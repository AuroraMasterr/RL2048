import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """DQN 网络"""
    
    def __init__(
        self,
        input_shape: tuple = (4, 4),
        num_actions: int = 4,
        hidden_dim: int = 256,
        use_one_hot: bool = True
    ):
        super().__init__()
        self.use_one_hot = use_one_hot
        if use_one_hot:
            self.input_size = input_shape[0] * input_shape[1] * 16
        else:
            self.input_size = input_shape[0] * input_shape[1]
        
        self.net = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_one_hot:
            x = self.to_one_hot(x)
        x = x.flatten(start_dim=1)
        return self.net(x)

    def to_one_hot(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        batch_size = x.shape[0]
        log_x = torch.zeros_like(x)
        mask = x > 0
        log_x[mask] = torch.log2(x[mask].float()).long()
        log_x = torch.clamp(log_x, 0, 15)
        one_hot = torch.zeros(batch_size, 4, 4, 16, device=x.device)
        one_hot.scatter_(3, log_x.unsqueeze(3), 1)
        return one_hot


class DuelingDQNNetwork(nn.Module):
    """Dueling DQN 网络"""
    
    def __init__(
        self,
        input_shape: tuple = (4, 4),
        num_actions: int = 4,
        hidden_dim: int = 256,
        use_one_hot: bool = True
    ):
        super().__init__()
        self.use_one_hot = use_one_hot
        if use_one_hot:
            self.input_size = input_shape[0] * input_shape[1] * 16
        else:
            self.input_size = input_shape[0] * input_shape[1]
        
        self.feature = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
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
        if self.use_one_hot:
            x = self.to_one_hot(x)
        x = x.flatten(start_dim=1)
        features = self.feature(x)
        advantage = self.advantage(features)
        value = self.value(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def to_one_hot(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        batch_size = x.shape[0]
        log_x = torch.zeros_like(x)
        mask = x > 0
        log_x[mask] = torch.log2(x[mask].float()).long()
        log_x = torch.clamp(log_x, 0, 15)
        one_hot = torch.zeros(batch_size, 4, 4, 16, device=x.device)
        one_hot.scatter_(3, log_x.unsqueeze(3), 1)
        return one_hot


class PPONetwork(nn.Module):
    """PPO 网络（Actor-Critic）- 支持One-hot编码"""
    
    def __init__(self, input_shape: tuple = (4, 4), num_actions: int = 4, hidden_dim: int = 256, use_one_hot: bool = True):
        super().__init__()
        self.use_one_hot = use_one_hot
        if use_one_hot:
            # One-hot编码：每个格子16个可能值（0-15）
            self.input_size = input_shape[0] * input_shape[1] * 16
        else:
            self.input_size = input_shape[0] * input_shape[1]
        
        # 更大的网络
        self.shared = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.actor = nn.Linear(hidden_dim // 2, num_actions)
        self.critic = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, x: torch.Tensor) -> tuple:
        if self.use_one_hot:
            x = self.to_one_hot(x)
        x = x.flatten(start_dim=1)
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value
    
    def to_one_hot(self, x: torch.Tensor) -> torch.Tensor:
        """将状态转换为One-hot编码 - 处理大数字"""
        # x的shape: (batch_size, 4, 4)
        x = x.long()
        batch_size = x.shape[0]
        
        # 将数字转换为对数尺度: 2^k -> k
        # 处理0值
        log_x = torch.zeros_like(x)
        mask = x > 0
        log_x[mask] = torch.log2(x[mask].float()).long()
        
        # 限制最大值为15（避免超出范围）
        log_x = torch.clamp(log_x, 0, 15)
        
        # 每个格子16个one-hot位
        one_hot = torch.zeros(batch_size, 4, 4, 16, device=x.device)
        # 填入对应位置的1
        one_hot.scatter_(3, log_x.unsqueeze(3), 1)
        # 返回(batch_size, 4, 4, 16)
        return one_hot


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
