import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class BaseAgent(ABC):
    """智能体基类"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    @abstractmethod
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """选择动作"""
        pass
    
    @abstractmethod
    def update(self, *args, **kwargs) -> dict:
        """更新智能体"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """保存模型"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """加载模型"""
        pass
    
    def preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """预处理状态：对数值取对数（可选）"""
        state = np.log2(state + 1)
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
