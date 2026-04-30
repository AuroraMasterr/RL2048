import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List

from .base_agent import BaseAgent
from models import PPONetwork


class PPOAgent(BaseAgent):
    """PPO 智能体 - 改进版"""
    
    def __init__(
        self,
        input_shape: tuple = (4, 4),
        num_actions: int = 4,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        use_one_hot: bool = True,
        device: str = 'cpu'
    ):
        super().__init__(device)
        
        self.num_actions = num_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        self.network = PPONetwork(input_shape, num_actions, hidden_dim, use_one_hot).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def select_action(self, state: np.ndarray, valid_actions: list = None, epsilon: float = 0.0) -> int:
        """选择动作 - 支持过滤无效动作"""
        with torch.no_grad():
            state_tensor = self.preprocess_state(state)
            logits, value = self.network(state_tensor)
            
            # 如果有有效动作列表，把无效动作的概率设为极小
            if valid_actions is not None and len(valid_actions) < 4:
                mask = torch.ones(self.num_actions, device=self.device) * -1e10
                for a in valid_actions:
                    mask[a] = 0
                logits = logits + mask
            
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool
    ) -> None:
        """存储转换"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_returns_and_advantages(self) -> tuple:
        """计算回报和优势"""
        returns = []
        advantages = []
        next_value = 0
        next_advantage = 0
        
        for reward, done, value in reversed(list(zip(self.rewards, self.dones, self.values))):
            if done:
                next_value = 0
                next_advantage = 0
            
            delta = reward + self.gamma * next_value * (1 - done) - value
            advantage = delta + self.gamma * self.gae_lambda * next_advantage * (1 - done)
            returns.append(advantage + value)
            advantages.append(advantage)
            
            next_value = value
            next_advantage = advantage
        
        returns = list(reversed(returns))
        advantages = list(reversed(advantages))
        
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update(self) -> Dict:
        """更新智能体"""
        if len(self.states) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0, 'total_loss': 0.0}
        
        returns, advantages = self.compute_returns_and_advantages()
        
        states = np.array(self.states)
        actions = np.array(self.actions)
        old_log_probs = np.array(self.log_probs)
        returns = np.array(returns)
        advantages = np.array(advantages)
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for _ in range(self.ppo_epochs):
            indices = np.arange(len(self.states))
            np.random.shuffle(indices)
            
            for start in range(0, len(self.states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # 直接用原始状态，网络会自己处理one-hot
                batch_states = torch.FloatTensor(states[batch_indices]).to(self.device)
                batch_actions = torch.LongTensor(actions[batch_indices]).to(self.device)
                batch_old_log_probs = torch.FloatTensor(old_log_probs[batch_indices]).to(self.device)
                batch_returns = torch.FloatTensor(returns[batch_indices]).to(self.device)
                batch_advantages = torch.FloatTensor(advantages[batch_indices]).to(self.device)
                
                logits, values = self.network(batch_states)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                entropy_loss = -dist.entropy().mean()
                
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        return {
            'policy_loss': np.mean(policy_losses) if policy_losses else 0.0,
            'value_loss': np.mean(value_losses) if value_losses else 0.0,
            'entropy_loss': np.mean(entropy_losses) if entropy_losses else 0.0,
            'total_loss': np.mean(policy_losses) + np.mean(value_losses) if policy_losses else 0.0
        }
    
    def save(self, path: str) -> None:
        """保存模型"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.network.eval()
