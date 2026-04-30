#!/usr/bin/env python3
"""改进版训练脚本 - 解决震荡和性能停滞问题"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import deque
import torch
import torch.optim as optim

from env import Game2048
from agents import PPOAgent
from config import Config

def evaluate_agent(agent, env, num_episodes=20):
    """评估智能体"""
    scores = []
    max_tiles = []
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            valid_actions = env.get_valid_actions()
            result = agent.select_action(state, epsilon=0.0)
            
            if isinstance(result, tuple) and len(result) >= 3:
                action, _, _ = result
            else:
                action = result
            
            if action not in valid_actions and valid_actions:
                action = np.random.choice(valid_actions)
            
            state, _, done, info = env.step(action)
        
        scores.append(info['score'])
        max_tiles.append(info['max_tile'])
    
    return np.mean(scores), np.mean(max_tiles), np.max(max_tiles)

def linear_lr_schedule(optimizer, initial_lr, final_lr, total_steps, current_step):
    """学习率线性衰减"""
    lr = initial_lr - (initial_lr - final_lr) * (current_step / total_steps)
    lr = max(lr, final_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

class AdvancedPPOTrainer:
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        self.env = Game2048(config.board_size)
        
        ppo_config = config.ppo.copy()
        if 'update_interval' in ppo_config:
            del ppo_config['update_interval']
        
        self.agent = PPOAgent(
            input_shape=(config.board_size, config.board_size),
            num_actions=config.num_actions,
            device=device,
            **ppo_config
        )
        
        # 高级优化器
        self.optimizer = optim.Adam(self.agent.network.parameters(), lr=ppo_config['lr'], eps=1e-5)
        
        # 最佳模型追踪
        self.best_score = 0
        self.no_improve_count = 0
        self.patience = 10  # 早停耐心值
        
        # 学习率调度
        self.initial_lr = ppo_config['lr']
        self.final_lr = self.initial_lr * 0.01
    
    def train_episode(self, epsilon=0.0):
        """训练一局"""
        state = self.env.reset()
        done = False
        episode_reward = 0
        transitions = []
        
        while not done:
            valid_actions = self.env.get_valid_actions()
            
            result = self.agent.select_action(state, epsilon=epsilon)
            if isinstance(result, tuple) and len(result) >= 3:
                action, log_prob, value = result
            else:
                action, log_prob = result
                value = 0
            
            if action not in valid_actions and valid_actions:
                action = np.random.choice(valid_actions)
            
            next_state, reward, done, info = self.env.step(action)
            
            transitions.append({
                'state': state,
                'action': action,
                'log_prob': log_prob,
                'value': value,
                'reward': reward,
                'done': done
            })
            
            state = next_state
            episode_reward += reward
        
        # 存储转换
        for trans in transitions:
            self.agent.store_transition(
                trans['state'],
                trans['action'],
                trans['log_prob'],
                trans['reward'],
                trans['value'],
                trans['done']
            )
        
        return info['score'], info['max_tile'], episode_reward
    
    def train(self, num_episodes=5000):
        """主训练循环"""
        scores_window = deque(maxlen=100)
        total_steps = 0
        
        # 探索率衰减
        initial_epsilon = 0.1
        final_epsilon = 0.01
        
        print("🚀 开始改进版训练...")
        print(f"  - 学习率: {self.initial_lr} -> {self.final_lr}")
        print(f"  - 探索率: {initial_epsilon} -> {final_epsilon}")
        print(f"  - 早停耐心: {self.patience} evaluations")
        
        for episode in tqdm(range(num_episodes), desc="Training"):
            # 动态调整探索率
            epsilon = initial_epsilon - (initial_epsilon - final_epsilon) * (episode / num_episodes)
            epsilon = max(epsilon, final_epsilon)
            
            # 训练一局
            score, max_tile, reward = self.train_episode(epsilon=epsilon)
            scores_window.append(score)
            total_steps += 1
            
            # 动态调整学习率
            linear_lr_schedule(self.optimizer, self.initial_lr, self.final_lr, num_episodes, episode)
            
            # 更新智能体（每2048步）
            update_interval = self.config.ppo.get('update_interval', 2048)
            if total_steps % update_interval == 0:
                update_info = self.agent.update()
            
            # 日志
            if (episode + 1) % self.config.log_interval == 0:
                avg_score = np.mean(scores_window)
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"📊 Episode {episode + 1}, "
                      f"Average Score (last 100): {avg_score:.2f}, "
                      f"LR: {current_lr:.6f}")
            
            # 评估
            if (episode + 1) % self.config.eval_interval == 0:
                eval_score, avg_max_tile, best_max_tile = evaluate_agent(self.agent, self.env, self.config.eval_episodes)
                print(f"🎯 Evaluation - "
                      f"Score: {eval_score:.2f}, "
                      f"Max Tile: {avg_max_tile:.2f}, "
                      f"Best: {best_max_tile}")
                
                # 保存最佳模型
                if eval_score > self.best_score:
                    self.best_score = eval_score
                    self.no_improve_count = 0
                    save_path = os.path.join(self.config.model_save_path, f'ppo_best_improved.pth')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    self.agent.save(save_path)
                    print(f"✅ New best model saved! Score: {eval_score:.2f}")
                else:
                    self.no_improve_count += 1
                    print(f"⏸️ No improvement for {self.no_improve_count} evaluations")
                
                # 早停
                if self.no_improve_count >= self.patience:
                    print(f"🛑 Early stopping triggered!")
                    break
            
            # 定期保存
            if (episode + 1) % self.config.save_interval == 0:
                save_path = os.path.join(self.config.model_save_path, f'ppo_episode_{episode + 1}.pth')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.agent.save(save_path)
        
        print(f"\n🎉 Training complete!")
        print(f"Best score: {self.best_score:.2f}")
        return self.best_score

def main():
    parser = argparse.ArgumentParser(description='Improved RL 2048 Training')
    parser.add_argument('--num_episodes', type=int, default=5000,
                       help='Number of episodes to train')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Computing device (cpu/cuda)')
    args = parser.parse_args()
    
    config = Config()
    trainer = AdvancedPPOTrainer(config, device=args.device)
    trainer.train(num_episodes=args.num_episodes)

if __name__ == '__main__':
    main()
