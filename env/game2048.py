import numpy as np
import random
from typing import Tuple, Optional


class Game2048:
    """2048 游戏环境"""
    
    def __init__(self, size: int = 4):
        self.size = size
        self.board = None
        self.score = 0
        self.reset()
    
    def reset(self) -> np.ndarray:
        """重置游戏"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self._add_new_tile()
        self._add_new_tile()
        return self._get_state()
    
    def _add_new_tile(self) -> None:
        """在随机空位添加 2 或 4"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            row, col = random.choice(empty_cells)
            self.board[row, col] = 2 if random.random() < 0.9 else 4
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        return self.board.copy()
    
    def _merge_left(self, row: np.ndarray) -> Tuple[np.ndarray, int]:
        """向左合并一行"""
        row = row[row != 0]
        score = 0
        i = 0
        while i < len(row) - 1:
            if row[i] == row[i + 1]:
                row[i] *= 2
                score += row[i]
                row = np.delete(row, i + 1)
            i += 1
        row = np.pad(row, (0, self.size - len(row)), mode='constant')
        return row, score
    
    def _move_left(self) -> Tuple[bool, int]:
        """向左移动"""
        moved = False
        total_score = 0
        for i in range(self.size):
            original = self.board[i].copy()
            new_row, score = self._merge_left(self.board[i])
            self.board[i] = new_row
            total_score += score
            if not np.array_equal(original, new_row):
                moved = True
        return moved, total_score
    
    def _move_right(self) -> Tuple[bool, int]:
        """向右移动"""
        self.board = np.fliplr(self.board)
        moved, score = self._move_left()
        self.board = np.fliplr(self.board)
        return moved, score
    
    def _move_up(self) -> Tuple[bool, int]:
        """向上移动"""
        self.board = self.board.T
        moved, score = self._move_left()
        self.board = self.board.T
        return moved, score
    
    def _move_down(self) -> Tuple[bool, int]:
        """向下移动"""
        self.board = self.board.T
        moved, score = self._move_right()
        self.board = self.board.T
        return moved, score
    
    def calculate_heuristic_reward(self, board, moved, score_gained):
        """计算启发式奖励 - 全面优化版本"""
        reward = 0
        
        if not moved:
            # 无效动作惩罚
            reward -= 5
            return reward
        
        # ========== 1. 合并得分奖励 ==========
        if score_gained > 0:
            # log2缩放，避免过大的分数掩盖其他奖励
            reward += np.log2(score_gained + 1) * 0.5
        
        # ========== 2. 空位数奖励（重要） ==========
        empty_cells = np.sum(board == 0)
        # 每多一个空位奖励更多（越后期越重要）
        reward += empty_cells * 1.0
        
        # ========== 3. 大数字在角落奖励 ==========
        max_tile = np.max(board)
        # 检查四个角落
        corner_positions = [(0, 0), (0, 3), (3, 0), (3, 3)]
        for i, j in corner_positions:
            if board[i, j] == max_tile:
                # 大数字在角落给重奖
                reward += np.log2(max_tile + 1) * 1.5
                break
        
        # ========== 4. 单调性奖励（从角落向外递增） ==========
        monotonicity = 0
        
        # 检查行（从左到右递增）
        for i in range(4):
            row = board[i, :]
            non_zero = row[row > 0]
            if len(non_zero) > 1:
                diffs = np.diff(non_zero)
                # 奖励递增的序列
                monotonicity += np.sum(diffs >= 0) * 0.3
                # 惩罚递减的序列
                monotonicity -= np.sum(diffs < 0) * 0.1
        
        # 检查列（从上到下递增）
        for j in range(4):
            col = board[:, j]
            non_zero = col[col > 0]
            if len(non_zero) > 1:
                diffs = np.diff(non_zero)
                monotonicity += np.sum(diffs >= 0) * 0.3
                monotonicity -= np.sum(diffs < 0) * 0.1
        
        reward += monotonicity
        
        # ========== 5. 平滑性奖励（相邻数字大小相近） ==========
        smoothness = 0
        for i in range(4):
            for j in range(4):
                val = board[i, j]
                if val > 0:
                    # 检查右边
                    if j < 3 and board[i, j + 1] > 0:
                        diff = abs(np.log2(val) - np.log2(board[i, j + 1]))
                        smoothness -= diff * 0.2
                    # 检查下边
                    if i < 3 and board[i + 1, j] > 0:
                        diff = abs(np.log2(val) - np.log2(board[i + 1, j]))
                        smoothness -= diff * 0.2
        
        reward += smoothness
        
        # ========== 6. 相同数字相邻奖励（容易合并） ==========
        same_neighbors = 0
        for i in range(4):
            for j in range(4):
                val = board[i, j]
                if val > 0:
                    # 检查右边
                    if j < 3 and board[i, j + 1] == val:
                        same_neighbors += np.log2(val + 1) * 0.3
                    # 检查下边
                    if i < 3 and board[i + 1, j] == val:
                        same_neighbors += np.log2(val + 1) * 0.3
        
        reward += same_neighbors
        
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, int, bool, dict]:
        """
        执行一步动作
        
        Args:
            action: 0=上, 1=下, 2=左, 3=右
        
        Returns:
            state, reward, done, info
        """
        if action == 0:
            moved, score = self._move_up()
        elif action == 1:
            moved, score = self._move_down()
        elif action == 2:
            moved, score = self._move_left()
        elif action == 3:
            moved, score = self._move_right()
        else:
            raise ValueError(f'无效动作: {action}')
        
        self.score += score
        
        # 计算启发式奖励
        reward = self.calculate_heuristic_reward(self.board, moved, score)
        
        if moved:
            self._add_new_tile()
        
        done = self._is_game_over()
        
        info = {
            'score': self.score,
            'max_tile': np.max(self.board),
            'moved': moved,
            'score_gained': score
        }
        
        return self._get_state(), reward, done, info
    
    def _is_game_over(self) -> bool:
        """检查游戏是否结束"""
        if np.any(self.board == 0):
            return False
        
        for i in range(self.size):
            for j in range(self.size):
                if j < self.size - 1 and self.board[i, j] == self.board[i, j + 1]:
                    return False
                if i < self.size - 1 and self.board[i, j] == self.board[i + 1, j]:
                    return False
        
        return True
    
    def get_valid_actions(self) -> list:
        """获取所有有效的动作"""
        valid_actions = []
        original_board = self.board.copy()
        
        for action in range(4):
            self.board = original_board.copy()
            if action == 0:
                moved, _ = self._move_up()
            elif action == 1:
                moved, _ = self._move_down()
            elif action == 2:
                moved, _ = self._move_left()
            elif action == 3:
                moved, _ = self._move_right()
            
            if moved:
                valid_actions.append(action)
        
        self.board = original_board
        return valid_actions
    
    def render(self) -> None:
        """渲染游戏界面"""
        print(f"Score: {self.score}")
        print("-" * (self.size * 6 + 1))
        for row in self.board:
            print("|", end="")
            for cell in row:
                if cell == 0:
                    print("     |", end="")
                else:
                    print(f"{cell:5d}|", end="")
            print()
            print("-" * (self.size * 6 + 1))
