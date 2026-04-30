#!/usr/bin/env python3
"""带可视化的改进版训练脚本"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import deque
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

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

class PPOTrainerWithPlots:
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
        
        self.optimizer = optim.Adam(self.agent.network.parameters(), lr=ppo_config['lr'], eps=1e-5)
        
        self.best_score = 0
        self.no_improve_count = 0
        self.patience = 10
        
        self.initial_lr = ppo_config['lr']
        self.final_lr = self.initial_lr * 0.01
        
        # 记录指标
        self.metrics = {
            'episodes': [],
            'avg_scores': [],
            'eval_scores': [],
            'eval_max_tiles': [],
            'best_max_tiles': [],
            'learning_rates': [],
            'epsilons': []
        }
    
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
        
        initial_epsilon = 0.1
        final_epsilon = 0.01
        
        print("🚀 开始改进版训练（带可视化）...")
        print(f"  - 学习率: {self.initial_lr} -> {self.final_lr}")
        print(f"  - 探索率: {initial_epsilon} -> {final_epsilon}")
        print(f"  - 早停耐心: {self.patience} evaluations")
        
        for episode in tqdm(range(num_episodes), desc="Training"):
            epsilon = initial_epsilon - (initial_epsilon - final_epsilon) * (episode / num_episodes)
            epsilon = max(epsilon, final_epsilon)
            
            score, max_tile, reward = self.train_episode(epsilon=epsilon)
            scores_window.append(score)
            total_steps += 1
            
            current_lr = linear_lr_schedule(self.optimizer, self.initial_lr, self.final_lr, num_episodes, episode)
            
            update_interval = self.config.ppo.get('update_interval', 2048)
            if total_steps % update_interval == 0:
                self.agent.update()
            
            if (episode + 1) % self.config.log_interval == 0:
                avg_score = np.mean(scores_window)
                print(f"📊 Episode {episode + 1}, "
                      f"Average Score (last 100): {avg_score:.2f}, "
                      f"LR: {current_lr:.6f}")
                
                # 记录指标
                self.metrics['episodes'].append(episode + 1)
                self.metrics['avg_scores'].append(avg_score)
                self.metrics['learning_rates'].append(current_lr)
                self.metrics['epsilons'].append(epsilon)
            
            if (episode + 1) % self.config.eval_interval == 0:
                eval_score, avg_max_tile, best_max_tile = evaluate_agent(self.agent, self.env, self.config.eval_episodes)
                print(f"🎯 Evaluation - "
                      f"Score: {eval_score:.2f}, "
                      f"Max Tile: {avg_max_tile:.2f}, "
                      f"Best: {best_max_tile}")
                
                # 记录指标
                self.metrics['eval_scores'].append(eval_score)
                self.metrics['eval_max_tiles'].append(avg_max_tile)
                self.metrics['best_max_tiles'].append(best_max_tile)
                
                if eval_score > self.best_score:
                    self.best_score = eval_score
                    self.no_improve_count = 0
                    save_path = os.path.join(self.config.model_save_path, f'ppo_best_plot.pth')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    self.agent.save(save_path)
                    print(f"✅ New best model saved! Score: {eval_score:.2f}")
                    
                    # 每次新纪录时画图
                    self.plot_metrics(intermediate=True)
                else:
                    self.no_improve_count += 1
                    print(f"⏸️ No improvement for {self.no_improve_count} evaluations")
                
                if self.no_improve_count >= self.patience:
                    print(f"🛑 Early stopping triggered!")
                    break
            
            if (episode + 1) % self.config.save_interval == 0:
                save_path = os.path.join(self.config.model_save_path, f'ppo_episode_{episode + 1}.pth')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.agent.save(save_path)
        
        print(f"\n🎉 Training complete!")
        print(f"Best score: {self.best_score:.2f}")
        
        # 最后画完整的图
        self.plot_metrics(intermediate=False)
        
        return self.best_score
    
    def plot_metrics(self, intermediate=False):
        """画出训练指标"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 平均得分
        ax = axes[0, 0]
        if self.metrics['avg_scores']:
            ax.plot(self.metrics['episodes'], self.metrics['avg_scores'], 'b-', linewidth=2, label='Avg Score (last 100)')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Score')
            ax.set_title('Training Progress - Average Score')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # 2. 评估得分和最大块
        ax = axes[0, 1]
        eval_episodes = self.metrics['episodes'][::2] if self.metrics['episodes'] else []
        if self.metrics['eval_scores']:
            ax2 = ax.twinx()
            line1, = ax.plot(eval_episodes[:len(self.metrics['eval_scores'])], self.metrics['eval_scores'], 'g-', linewidth=2, label='Eval Score')
            line2, = ax2.plot(eval_episodes[:len(self.metrics['eval_max_tiles'])], self.metrics['eval_max_tiles'], 'r-', linewidth=2, label='Avg Max Tile')
            line3, = ax2.plot(eval_episodes[:len(self.metrics['best_max_tiles'])], self.metrics['best_max_tiles'], 'o', markersize=6, color='orange', label='Best Max Tile')
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Score', color='g')
            ax2.set_ylabel('Max Tile', color='r')
            ax.set_title('Evaluation Performance')
            ax.grid(True, alpha=0.3)
            ax.legend(handles=[line1, line2, line3], loc='upper left')
        
        # 3. 学习率和探索率
        ax = axes[1, 0]
        if self.metrics['learning_rates']:
            ax2 = ax.twinx()
            line1, = ax.plot(self.metrics['episodes'], self.metrics['learning_rates'], 'b-', linewidth=2, label='Learning Rate')
            line2, = ax2.plot(self.metrics['episodes'], self.metrics['epsilons'], 'm-', linewidth=2, label='Epsilon')
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Learning Rate', color='b')
            ax2.set_ylabel('Epsilon', color='m')
            ax.set_title('Learning Rate and Epsilon Schedule')
            ax.grid(True, alpha=0.3)
            ax.legend(handles=[line1, line2], loc='upper right')
        
        # 4. 最大块分布
        ax = axes[1, 1]
        if self.metrics['best_max_tiles']:
            tiles = np.array(self.metrics['best_max_tiles'])
            bins = [16, 32, 64, 128, 256, 512, 1024, 2048]
            ax.hist(tiles, bins=bins, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Max Tile')
            ax.set_ylabel('Count')
            ax.set_title('Max Tile Distribution')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log', base=2)
            ax.set_xticks(bins)
            ax.set_xticklabels([str(b) for b in bins])
        
        plt.tight_layout()
        
        if intermediate:
            save_path = os.path.join('logs', f'training_progress_ep{self.metrics["episodes"][-1]}.png')
        else:
            save_path = os.path.join('logs', 'training_progress_final.png')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Improved RL 2048 Training with Plots')
    parser.add_argument('--num_episodes', type=int, default=5000,
                       help='Number of episodes to train')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Computing device (cpu/cuda)')
    args = parser.parse_args()
    
    config = Config()
    trainer = PPOTrainerWithPlots(config, device=args.device)
    trainer.train(num_episodes=args.num_episodes)

if __name__ == '__main__':
    main()
