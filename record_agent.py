#!/usr/bin/env python3
"""
录制 PPO 智能体玩 2048 的视频
使用 Python 直接生成视频，不依赖浏览器
"""

import os
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from env import Game2048
from agents import PPOAgent
from config import Config

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 游戏和颜色设置
TILE_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}

TEXT_COLORS = {
    2: (119, 110, 101),
    4: (119, 110, 101),
    8: (249, 246, 242),
    16: (249, 246, 242),
    32: (249, 246, 242),
    64: (249, 246, 242),
    128: (249, 246, 242),
    256: (249, 246, 242),
    512: (249, 246, 242),
    1024: (249, 246, 242),
    2048: (249, 246, 242),
}

ACTION_NAMES = ['上', '下', '左', '右']


def draw_board(board, score, max_tile, step, action=None):
    """绘制游戏棋盘"""
    # 创建图像
    img_size = 600
    img = Image.new('RGB', (img_size, img_size), (250, 248, 239))
    draw = ImageDraw.Draw(img)
    
    # 绘制标题和信息 - 使用英文避免字体问题
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Arial Bold.ttf", 36)
        font_info = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        font_num = ImageFont.truetype("/System/Library/Fonts/Arial Bold.ttf", 32)
    except:
        # 回退到默认字体
        font_title = ImageFont.load_default()
        font_info = ImageFont.load_default()
        font_num = ImageFont.load_default()
    
    # 绘制标题
    draw.text((20, 20), "2048 - PPO Agent", fill=(119, 110, 101), font=font_title)
    
    # 绘制信息 - 使用英文
    info_text = f"Score: {score} | Max: {max_tile} | Step: {step}"
    if action is not None:
        action_names = ['Up', 'Down', 'Left', 'Right']
        info_text += f" | Action: {action_names[action]}"
    draw.text((20, 70), info_text, fill=(119, 110, 101), font=font_info)
    
    # 绘制棋盘背景
    board_size = 480
    cell_size = board_size // 4
    board_x = (img_size - board_size) // 2
    board_y = 120
    draw.rectangle([board_x, board_y, board_x + board_size, board_y + board_size], 
                  fill=(187, 173, 160), outline=(160, 149, 137), width=3)
    
    # 绘制格子和数字
    for i in range(4):
        for j in range(4):
            cell_x = board_x + j * cell_size + 8
            cell_y = board_y + i * cell_size + 8
            cell_width = cell_size - 16
            
            value = board[i][j]
            color = TILE_COLORS.get(value, TILE_COLORS[0])
            
            # 绘制格子
            draw.rectangle([cell_x, cell_y, cell_x + cell_width, cell_y + cell_width], 
                          fill=color, outline=None)
            
            # 绘制数字
            if value > 0:
                text_color = TEXT_COLORS.get(value, (119, 110, 101))
                text = str(value)
                
                # 计算文本位置（居中）
                bbox = draw.textbbox((0, 0), text, font=font_num)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                text_x = cell_x + (cell_width - text_width) // 2
                text_y = cell_y + (cell_width - text_height) // 2
                
                draw.text((text_x, text_y), text, fill=text_color, font=font_num)
    
    return img


def record_agent_video(model_path=None, num_episodes=3, max_steps=1000, fps=2, output_dir="videos"):
    """录制智能体玩游戏的视频"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载智能体
    print("加载智能体...")
    config = Config()
    
    # 过滤掉 PPOAgent 不支持的参数
    ppo_config = config.ppo.copy()
    if 'update_interval' in ppo_config:
        del ppo_config['update_interval']
    
    agent = PPOAgent(
        input_shape=(config.board_size, config.board_size),
        num_actions=config.num_actions,
        device='cpu',
        **ppo_config
    )
    
    # 尝试加载模型
    if model_path is None:
        model_path = "models/checkpoints/ppo_best.pth"
    
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"成功加载模型: {model_path}")
    else:
        print(f"警告: 模型文件不存在: {model_path}")
        print("将使用随机动作演示...")
        agent = None
    
    # 录制游戏
    frames = []
    all_scores = []
    all_max_tiles = []
    invalid_move_count = 0
    
    for episode in range(num_episodes):
        print(f"\n=== 游戏 {episode + 1}/{num_episodes} ===")
        
        game = Game2048()
        state = game.board
        done = False
        step = 0
        frame_step = 0
        
        # 绘制初始状态
        img = draw_board(state, game.score, np.max(state), frame_step)
        frames.append(img)
        
        while not done and step < max_steps:
            # 获取有效动作
            valid_actions = game.get_valid_actions()
            
            # 选择动作 - 优先从有效动作中选择
            if agent is not None:
                result = agent.select_action(state, epsilon=0.0)
                if isinstance(result, tuple) and len(result) >= 3:
                    action, _, _ = result
                elif isinstance(result, tuple) and len(result) == 2:
                    action, _ = result
                else:
                    action = result
                
                # 如果动作无效，从有效动作中随机选择一个
                if action not in valid_actions and valid_actions:
                    action = np.random.choice(valid_actions)
                    invalid_move_count += 1
            else:
                if valid_actions:
                    action = np.random.choice(valid_actions)
                else:
                    action = np.random.choice(4)
            
            # 执行动作
            state, reward, done, info = game.step(action)
            step += 1
            
            # 只在确实移动了的时候增加帧（避免看起来卡顿）
            if info['moved']:
                frame_step += 1
                img = draw_board(state, game.score, info['max_tile'], frame_step, action)
                frames.append(img)
            
            if step % 10 == 0:
                print(f"  步数: {step}, 分数: {game.score}, 最大块: {info['max_tile']}")
        
        print(f"游戏结束! 分数: {game.score}, 最大块: {info['max_tile']}, 总步数: {step}")
        all_scores.append(game.score)
        all_max_tiles.append(info['max_tile'])
    
    # 打印统计信息
    print(f"\n=== 统计信息 ===")
    print(f"平均分数: {np.mean(all_scores):.2f}")
    print(f"最高分数: {np.max(all_scores)}")
    print(f"平均最大块: {np.mean(all_max_tiles):.2f}")
    print(f"最高最大块: {np.max(all_max_tiles)}")
    print(f"修正的无效动作次数: {invalid_move_count}")
    
    # 保存视频
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"2048_ppo_agent_{timestamp}.mp4")
    
    print(f"\n保存视频到: {output_path}")
    
    # 尝试使用 imageio 保存 MP4
    try:
        import imageio
        writer = imageio.get_writer(output_path, fps=fps)
        
        for frame in frames:
            # 转换 PIL 图像为 numpy 数组
            frame_np = np.array(frame)
            writer.append_data(frame_np)
        
        writer.close()
        print(f"✅ 视频保存成功!")
        
    except ImportError:
        print("警告: 未安装 imageio，尝试使用 PIL 保存为 GIF...")
        try:
            # 保存为 GIF
            output_path_gif = output_path.replace('.mp4', '.gif')
            frames[0].save(
                output_path_gif,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000/fps),
                loop=0
            )
            print(f"✅ GIF 保存成功: {output_path_gif}")
            print(f"提示: 如需 MP4 格式，请安装: pip install imageio imageio-ffmpeg")
        except Exception as e:
            print(f"保存 GIF 失败: {e}")
            # 最后尝试保存为图片序列
            seq_dir = os.path.join(output_dir, f"frames_{timestamp}")
            os.makedirs(seq_dir, exist_ok=True)
            for i, frame in enumerate(frames):
                frame.save(os.path.join(seq_dir, f"frame_{i:05d}.png"))
            print(f"✅ 图片序列保存成功: {seq_dir}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="录制 PPO 智能体玩 2048 的视频")
    parser.add_argument("--model", type=str, default=None, help="模型路径")
    parser.add_argument("--episodes", type=int, default=1, help="游戏次数")
    parser.add_argument("--fps", type=int, default=3, help="视频帧率")
    parser.add_argument("--output", type=str, default="videos", help="输出目录")
    
    args = parser.parse_args()
    
    record_agent_video(
        model_path=args.model,
        num_episodes=args.episodes,
        fps=args.fps,
        output_dir=args.output
    )
