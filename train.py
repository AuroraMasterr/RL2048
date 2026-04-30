#!/usr/bin/env python3
"""训练脚本"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import deque

from env import Game2048
from agents import DQNAgent, PPOAgent
from config import Config


def parse_args():
    parser = argparse.ArgumentParser(description='RL 2048 Training')
    parser.add_argument('--algorithm', type=str, default='ppo',
                        choices=['dqn', 'double_dqn', 'dueling_dqn', 'ppo'],
                        help='选择强化学习算法')
    parser.add_argument('--num_episodes', type=int, default=10000,
                        help='训练回合数')
    parser.add_argument('--device', type=str, default='cpu',
                        help='计算设备 (cpu/cuda)')
    parser.add_argument('--save_dir', type=str, default='./models/checkpoints',
                        help='模型保存目录')
    return parser.parse_args()


def evaluate_agent(agent, env, num_episodes=10):
    """评估智能体"""
    scores = []
    max_tiles = []
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # 获取有效动作
            valid_actions = env.get_valid_actions()
            
            if hasattr(agent, 'select_action') and callable(getattr(agent, 'select_action')):
                result = agent.select_action(state, epsilon=0.0)
                if isinstance(result, tuple) and len(result) >= 3:
                    action, _, _ = result
                elif isinstance(result, tuple) and len(result) == 2:
                    action, _ = result
                else:
                    action = result
            else:
                action = agent.select_action(state, epsilon=0.0)
            
            # 如果动作无效，从有效动作中随机选
            if action not in valid_actions and valid_actions:
                action = np.random.choice(valid_actions)
            
            state, reward, done, info = env.step(action)
            episode_reward += reward
        
        scores.append(info['score'])
        max_tiles.append(info['max_tile'])
    
    return np.mean(scores), np.mean(max_tiles), np.max(max_tiles)


def train_dqn_agent(config, args):
    """训练 DQN 类智能体"""
    env = Game2048(config.board_size)
    
    if args.algorithm == 'dqn':
        agent = DQNAgent(
            input_shape=(config.board_size, config.board_size),
            num_actions=config.num_actions,
            double_dqn=False,
            dueling=False,
            device=args.device,
            **config.dqn
        )
    elif args.algorithm == 'double_dqn':
        agent = DQNAgent(
            input_shape=(config.board_size, config.board_size),
            num_actions=config.num_actions,
            double_dqn=True,
            dueling=False,
            device=args.device,
            **config.double_dqn
        )
    else:  # dueling_dqn
        agent = DQNAgent(
            input_shape=(config.board_size, config.board_size),
            num_actions=config.num_actions,
            double_dqn=True,
            dueling=True,
            device=args.device,
            **config.dueling_dqn
        )
    
    scores_window = deque(maxlen=100)
    best_score = 0
    
    for episode in tqdm(range(args.num_episodes), desc=f'Training {args.algorithm}'):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # 获取有效动作
            valid_actions = env.get_valid_actions()
            
            action = agent.select_action(state)
            
            # 如果动作无效，从有效动作中随机选
            if action not in valid_actions and valid_actions:
                action = np.random.choice(valid_actions)
            
            next_state, reward, done, info = env.step(action)
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            update_info = agent.update()
            
            state = next_state
            episode_reward += reward
        
        scores_window.append(info['score'])
        
        if (episode + 1) % config.log_interval == 0:
            avg_score = np.mean(scores_window)
            print(f'Episode {episode + 1}, Average Score (last 100): {avg_score:.2f}, '
                  f'Epsilon: {update_info.get("epsilon", 0):.4f}')
        
        if (episode + 1) % config.eval_interval == 0:
            eval_score, avg_max_tile, max_tile = evaluate_agent(agent, env, config.eval_episodes)
            print(f'Evaluation - Average Score: {eval_score:.2f}, '
                  f'Average Max Tile: {avg_max_tile:.2f}, '
                  f'Best Max Tile: {max_tile}')
            
            if eval_score > best_score:
                best_score = eval_score
                save_path = os.path.join(args.save_dir, f'{args.algorithm}_best.pth')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                agent.save(save_path)
                print(f'Model saved: {save_path}')
        
        if (episode + 1) % config.save_interval == 0:
            save_path = os.path.join(args.save_dir, f'{args.algorithm}_episode_{episode + 1}.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            agent.save(save_path)


def train_ppo_agent(config, args):
    """训练 PPO 智能体"""
    env = Game2048(config.board_size)
    
    # 创建 PPO 配置，移除 update_interval
    ppo_config = config.ppo.copy()
    update_interval = ppo_config.pop('update_interval')
    
    agent = PPOAgent(
        input_shape=(config.board_size, config.board_size),
        num_actions=config.num_actions,
        device=args.device,
        **ppo_config
    )
    
    scores_window = deque(maxlen=100)
    best_score = 0
    total_steps = 0
    
    for episode in tqdm(range(args.num_episodes), desc='Training PPO'):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # 获取有效动作
            valid_actions = env.get_valid_actions()
            
            action, log_prob, value = agent.select_action(state)
            
            # 如果动作无效，从有效动作中随机选
            if action not in valid_actions and valid_actions:
                action = np.random.choice(valid_actions)
            
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, log_prob, reward, value, done)
            
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            if total_steps % update_interval == 0:
                update_info = agent.update()
        
        scores_window.append(info['score'])
        
        if (episode + 1) % config.log_interval == 0:
            avg_score = np.mean(scores_window)
            print(f'Episode {episode + 1}, Average Score (last 100): {avg_score:.2f}')
        
        if (episode + 1) % config.eval_interval == 0:
            eval_score, avg_max_tile, max_tile = evaluate_agent(agent, env, config.eval_episodes)
            print(f'Evaluation - Average Score: {eval_score:.2f}, '
                  f'Average Max Tile: {avg_max_tile:.2f}, '
                  f'Best Max Tile: {max_tile}')
            
            if eval_score > best_score:
                best_score = eval_score
                save_path = os.path.join(args.save_dir, 'ppo_best.pth')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                agent.save(save_path)
                print(f'Model saved: {save_path}')
        
        if (episode + 1) % config.save_interval == 0:
            save_path = os.path.join(args.save_dir, f'ppo_episode_{episode + 1}.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            agent.save(save_path)


def main():
    args = parse_args()
    config = Config()
    
    print(f'Starting training with algorithm: {args.algorithm}')
    print(f'Number of episodes: {args.num_episodes}')
    print(f'Device: {args.device}')
    
    if args.algorithm in ['dqn', 'double_dqn', 'dueling_dqn']:
        train_dqn_agent(config, args)
    elif args.algorithm == 'ppo':
        train_ppo_agent(config, args)
    else:
        raise ValueError(f'Unknown algorithm: {args.algorithm}')


if __name__ == '__main__':
    main()
