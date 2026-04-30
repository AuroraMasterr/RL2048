import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """DQN 网络"""
    
    def __init__(self, input_shape: tuple = (4, 4), num_actions: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.input_size = input_shape[0] * input_shape[1]
        
        self.fc1 = nn.Linear(self.input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingDQNNetwork(nn.Module):
    """Dueling DQN 网络"""
    
    def __init__(self, input_shape: tuple = (4, 4), num_actions: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.input_size = input_shape[0] * input_shape[1]
        
        self.feature = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        features = self.feature(x)
        advantage = self.advantage(features)
        value = self.value(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class PPONetwork(nn.Module):
    """PPO 网络（Actor-Critic）"""
    
    def __init__(self, input_shape: tuple = (4, 4), num_actions: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.input_size = input_shape[0] * input_shape[1]
        
        self.shared = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> tuple:
        x = x.flatten(start_dim=1)
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value


class A2CNetwork(nn.Module):
    """A2C 网络（Actor-Critic）"""
    
    def __init__(self, input_shape: tuple = (4, 4), num_actions: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.input_size = input_shape[0] * input_shape[1]
        
        self.shared = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> tuple:
        x = x.flatten(start_dim=1)
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value


class SACNetwork(nn.Module):
    """SAC 网络"""
    
    def __init__(self, input_shape: tuple = (4, 4), num_actions: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.input_size = input_shape[0] * input_shape[1]
        
        self.policy = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
        self.q1 = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward_policy(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        return self.policy(x)
    
    def forward_q1(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        return self.q1(x)
    
    def forward_q2(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        return self.q2(x)
