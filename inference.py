#!/usr/bin/env python3
"""推理脚本"""

import argparse
import numpy as np
from tqdm import tqdm

from env import Game2048
from agents import DQNAgent, PPOAgent
from config import Config


def parse_args():
    parser = argparse.ArgumentParser(description='RL 2048 Inference')
    parser.add_argument('--algorithm', type=str, default='ppo',
                        choices=['dqn', 'double_dqn', 'dueling_dqn', 'ppo'],
                        help='选择强化学习算法')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型文件路径')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='测试回合数')
    parser.add_argument('--device', type=str, default='cpu',
                        help='计算设备 (cpu/cuda)')
    parser.add_argument('--render', action='store_true',
                        help='渲染游戏过程')
    return parser.parse_args()


def load_agent(algorithm, model_path, config, device='cpu'):
    """加载智能体"""
    if algorithm in ['dqn', 'double_dqn', 'dueling_dqn']:
        double_dqn = algorithm != 'dqn'
        dueling = algorithm == 'dueling_dqn'
        
        agent = DQNAgent(
            input_shape=(config.board_size, config.board_size),
            num_actions=config.num_actions,
            double_dqn=double_dqn,
            dueling=dueling,
            device=device,
            **config.dqn
        )
    elif algorithm == 'ppo':
        ppo_config = config.ppo.copy()
        ppo_config.pop('update_interval', None)
        
        agent = PPOAgent(
            input_shape=(config.board_size, config.board_size),
            num_actions=config.num_actions,
            device=device,
            **ppo_config
        )
    else:
        raise ValueError(f'Unknown algorithm: {algorithm}')
    
    agent.load(model_path)
    return agent


def run_episode(agent, env, render=False):
    """运行一个回合"""
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    if render:
        env.render()
    
    while not done:
        if hasattr(agent, 'select_action'):
            result = agent.select_action(state, epsilon=0.0)
            if isinstance(result, tuple) and len(result) >= 3:
                action, _, _ = result
            elif isinstance(result, tuple) and len(result) == 2:
                action, _ = result
            else:
                action = result
        else:
            action = agent.select_action(state, epsilon=0.0)
        
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if render:
            env.render()
    
    return info['score'], info['max_tile'], steps


def main():
    args = parse_args()
    config = Config()
    
    print(f'Loading {args.algorithm} agent from {args.model_path}')
    agent = load_agent(args.algorithm, args.model_path, config, args.device)
    
    env = Game2048(config.board_size)
    
    scores = []
    max_tiles = []
    all_steps = []
    
    print(f'Running {args.num_episodes} episodes...')
    
    for episode in tqdm(range(args.num_episodes)):
        score, max_tile, steps = run_episode(agent, env, args.render)
        scores.append(score)
        max_tiles.append(max_tile)
        all_steps.append(steps)
    
    print('\n' + '=' * 50)
    print('Inference Results:')
    print('=' * 50)
    print(f'Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}')
    print(f'Median Score: {np.median(scores):.2f}')
    print(f'Max Score: {np.max(scores)}')
    print(f'Min Score: {np.min(scores)}')
    print()
    print(f'Average Max Tile: {np.mean(max_tiles):.2f} ± {np.std(max_tiles):.2f}')
    print(f'Median Max Tile: {np.median(max_tiles)}')
    print(f'Best Max Tile: {np.max(max_tiles)}')
    print()
    print(f'Average Steps: {np.mean(all_steps):.2f}')
    
    # 统计达到不同瓦片的次数
    tile_counts = {}
    for tile in max_tiles:
        tile_counts[tile] = tile_counts.get(tile, 0) + 1
    
    print('\nMax Tile Distribution:')
    for tile in sorted(tile_counts.keys()):
        print(f'  {tile}: {tile_counts[tile]} times ({tile_counts[tile]/args.num_episodes*100:.1f}%)')


if __name__ == '__main__':
    main()
