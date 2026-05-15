#!/usr/bin/env python3
"""2048 游戏 Flask 应用"""

import os
import sys

import numpy as np
from flask import Flask, jsonify, render_template, request

from env import Game2048

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# 存储游戏实例
game_instances = {}
agent_instance = None
agent_type = None


def get_agent():
    """获取智能体实例"""
    global agent_instance, agent_type
    
    if agent_instance is None:
        try:
            from agents import DQNAgent, PPOAgent
            from config import Config
            
            config = Config()
            dqn_model_path = 'models/checkpoints/dqn_best.pth'
            ppo_model_path = 'models/checkpoints/ppo_best_improved_v2.pth'

            if os.path.exists(dqn_model_path):
                dqn_config = config.dqn.copy()
                agent_instance = DQNAgent(
                    input_shape=(config.board_size, config.board_size),
                    num_actions=config.num_actions,
                    device='cpu',
                    **dqn_config
                )
                agent_instance.load(dqn_model_path)
                agent_type = 'dqn'
                print(f"成功加载 DQN 智能体模型: {dqn_model_path}")
            elif os.path.exists(ppo_model_path):
                ppo_config = config.ppo.copy()
                ppo_config.pop('update_interval', None)
                ppo_config['hidden_dim'] = 256
                agent_instance = PPOAgent(
                    input_shape=(config.board_size, config.board_size),
                    num_actions=config.num_actions,
                    use_one_hot=True,
                    device='cpu',
                    **ppo_config
                )
                agent_instance.load(ppo_model_path)
                agent_type = 'ppo'
                print(f"成功加载 PPO 智能体模型: {ppo_model_path}")
            else:
                print(f"模型文件不存在: {dqn_model_path} / {ppo_model_path}")
        except Exception as e:
            print(f"加载智能体失败: {e}")
    
    return agent_instance


@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/demo')
def demo():
    """智能体演示页面"""
    return render_template('demo.html')


@app.route('/api/reset', methods=['POST'])
def reset():
    """重置游戏"""
    game_id = request.json.get('game_id', 'default')
    game = Game2048()
    game_instances[game_id] = game
    state = game.board.tolist()
    return jsonify({
        'board': state,
        'score': int(game.score),
        'max_tile': int(np.max(game.board))
    })


@app.route('/api/step', methods=['POST'])
def step():
    """执行一步动作"""
    game_id = request.json.get('game_id', 'default')
    action = request.json.get('action')
    
    if game_id not in game_instances:
        return jsonify({'error': 'Game not found'}), 404
    
    game = game_instances[game_id]
    state, reward, done, info = game.step(action)
    
    return jsonify({
        'board': state.tolist(),
        'score': int(game.score),
        'max_tile': int(info['max_tile']),
        'done': done,
        'reward': int(reward)
    })


@app.route('/api/agent/step', methods=['POST'])
def agent_step():
    """智能体执行一步"""
    game_id = request.json.get('game_id', 'default')
    
    if game_id not in game_instances:
        # 创建新游戏
        game = Game2048()
        game_instances[game_id] = game
    
    game = game_instances[game_id]
    
    # 获取智能体动作
    agent = get_agent()
    if agent is None:
        return jsonify({'error': 'Agent not available'}), 500
    
    state = game.board
    action = None
    valid_actions = game.get_valid_actions()
    
    try:
        if agent_type == 'ppo':
            result = agent.select_action(state, valid_actions=valid_actions, epsilon=0.0)
            if isinstance(result, tuple) and len(result) >= 3:
                action, _, _ = result
            elif isinstance(result, tuple) and len(result) == 2:
                action, _ = result
            else:
                action = result
        else:
            action = agent.select_action(state, valid_actions=valid_actions, epsilon=0.0)
    except Exception as e:
        print(f"智能体决策错误: {e}")
        return jsonify({'error': str(e)}), 500
    
    # 执行动作
    next_state, reward, done, info = game.step(action)
    
    return jsonify({
        'board': next_state.tolist(),
        'score': int(game.score),
        'max_tile': int(info['max_tile']),
        'done': done,
        'action': int(action),
        'reward': float(reward),
        'agent_type': agent_type
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
