import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Optional, Dict

from .base_agent import BaseAgent
from models import DQNNetwork, DuelingDQNNetwork
from utils import ReplayBuffer


class DQNAgent(BaseAgent):
    """DQN 智能体"""
    
    def __init__(
        self,
        input_shape: tuple = (4, 4),
        num_actions: int = 4,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update: int = 100,
        buffer_size: int = 10000,
        batch_size: int = 32,
        double_dqn: bool = False,
        dueling: bool = False,
        device: str = 'cpu'
    ):
        super().__init__(device)
        
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.batch_size = batch_size
        self.double_dqn = double_dqn
        self.dueling = dueling
        
        if dueling:
            self.policy_net = DuelingDQNNetwork(input_shape, num_actions, hidden_dim).to(self.device)
            self.target_net = DuelingDQNNetwork(input_shape, num_actions, hidden_dim).to(self.device)
        else:
            self.policy_net = DQNNetwork(input_shape, num_actions, hidden_dim).to(self.device)
            self.target_net = DQNNetwork(input_shape, num_actions, hidden_dim).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.step_count = 0
    
    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """选择动作"""
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        
        with torch.no_grad():
            state_tensor = self.preprocess_state(state)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def update(self) -> Dict:
        """更新智能体"""
        if len(self.replay_buffer) < self.batch_size:
            return {'loss': 0.0}
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states_tensor = torch.FloatTensor(np.log2(states + 1)).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(np.log2(next_states + 1)).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        current_q_values = self.policy_net(states_tensor).gather(1, actions_tensor)
        
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.policy_net(next_states_tensor).argmax(1, keepdim=True)
                next_q_values = self.target_net(next_states_tensor).gather(1, next_actions)
            else:
                next_q_values = self.target_net(next_states_tensor).max(1, keepdim=True)[0]
            
            target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values
        
        loss = self.criterion(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return {'loss': loss.item(), 'epsilon': self.epsilon}
    
    def save(self, path: str) -> None:
        """保存模型"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, path)
    
    def load(self, path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.step_count = checkpoint.get('step_count', 0)
        self.target_net.eval()
