import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Optional, Dict

from .base_agent import BaseAgent
from models import DQNNetwork, DuelingDQNNetwork
from utils import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent(BaseAgent):
    """DQN 智能体"""
    
    def __init__(
        self,
        input_shape: tuple = (4, 4),
        num_actions: int = 4,
        hidden_dim: int = 256,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.997,
        target_update: int = 1,
        buffer_size: int = 50000,
        batch_size: int = 128,
        double_dqn: bool = False,
        dueling: bool = False,
        prioritized_replay: bool = True,
        priority_alpha: float = 0.6,
        priority_beta: float = 0.4,
        tau: float = 0.01,
        grad_clip_norm: float = 10.0,
        use_one_hot: bool = True,
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
        self.prioritized_replay = prioritized_replay
        self.priority_beta = priority_beta
        self.tau = tau
        self.grad_clip_norm = grad_clip_norm
        
        if dueling:
            self.policy_net = DuelingDQNNetwork(input_shape, num_actions, hidden_dim, use_one_hot).to(self.device)
            self.target_net = DuelingDQNNetwork(input_shape, num_actions, hidden_dim, use_one_hot).to(self.device)
        else:
            self.policy_net = DQNNetwork(input_shape, num_actions, hidden_dim, use_one_hot).to(self.device)
            self.target_net = DQNNetwork(input_shape, num_actions, hidden_dim, use_one_hot).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss(reduction='none')
        
        if prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=priority_alpha, beta=priority_beta)
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)
        self.step_count = 0
    
    def select_action(
        self,
        state: np.ndarray,
        valid_actions: Optional[list] = None,
        epsilon: Optional[float] = None
    ) -> int:
        """选择动作"""
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            if valid_actions:
                return random.choice(valid_actions)
            return random.randint(0, self.num_actions - 1)
        
        with torch.no_grad():
            state_tensor = self.preprocess_state(state)
            q_values = self.policy_net(state_tensor)
            if valid_actions is not None and len(valid_actions) < self.num_actions:
                mask = torch.full_like(q_values, -1e10)
                mask[:, valid_actions] = 0
                q_values = q_values + mask
            return q_values.argmax().item()
    
    def update(self) -> Dict:
        """更新智能体"""
        if len(self.replay_buffer) < self.batch_size:
            return {'loss': 0.0}
        
        if self.prioritized_replay:
            states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)
            weights_tensor = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            indices = None
            weights_tensor = torch.ones((len(states), 1), device=self.device)

        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        current_q_values = self.policy_net(states_tensor).gather(1, actions_tensor)
        
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.policy_net(next_states_tensor).argmax(1, keepdim=True)
                next_q_values = self.target_net(next_states_tensor).gather(1, next_actions)
            else:
                next_q_values = self.target_net(next_states_tensor).max(1, keepdim=True)[0]
            
            target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values
        
        td_errors = target_q_values - current_q_values
        loss = (self.criterion(current_q_values, target_q_values) * weights_tensor).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        if self.prioritized_replay and indices is not None:
            new_priorities = td_errors.detach().abs().cpu().numpy().flatten() + 1e-6
            self.replay_buffer.update_priorities(indices, new_priorities)
        
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(
                    self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
                )
        
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
