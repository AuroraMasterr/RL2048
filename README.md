# RL2048 - 强化学习训练 2048 游戏

使用强化学习算法训练智能体玩 2048 游戏的 Python 项目。

## 项目结构

```
RL2048/
├── env/              # 2048 游戏环境
├── agents/           # 智能体实现（多种算法）
├── models/           # 神经网络模型
├── config/           # 配置文件
├── utils/            # 工具函数
├── logs/             # 训练日志
├── train.py          # 训练脚本
├── inference.py      # 推理脚本
├── requirements.txt  # 依赖库
└── README.md         # 项目说明
```

## 强化学习算法方案对比

### 方案一：DQN (Deep Q-Network)

**核心思想**：使用神经网络近似 Q 函数，通过经验回放和目标网络稳定训练。

**优势**：
- 实现相对简单，易于理解和调试
- 有丰富的理论基础和实践经验
- 离散动作空间效果良好（2048 正好是 4 个动作）
- 计算资源要求适中

**劣势**：
- 样本效率较低，需要大量训练
- 超参数敏感
- 难以处理部分可观测性
- Q 值过估计问题

**适用场景**：作为基准算法快速验证

---

### 方案二：Double DQN

**核心思想**：使用两个网络分别选择动作和评估 Q 值，解决过估计问题。

**优势**：
- 改善 DQN 的 Q 值过估计问题
- 训练更稳定
- 只需要在 DQN 基础上做小修改
- 收敛速度更快

**劣势**：
- 仍然存在样本效率问题
- 需要维护两个网络

**适用场景**：需要比 DQN 更好性能的场景

---

### 方案三：Dueling DQN

**核心思想**：将 Q 函数分解为状态价值函数和优势函数，更好地学习状态价值。

**优势**：
- 能够更好地评估状态的重要性
- 在某些任务上显著提升性能
- 可以与 Double DQN 结合使用
- 网络结构改动不大

**劣势**：
- 网络结构稍复杂
- 对于简单任务提升不明显

**适用场景**：需要精细状态评估的任务

---

### 方案四：Rainbow DQN

**核心思想**：结合多种改进技巧（Double DQN、Prioritized Experience Replay、Dueling Network、Multi-step Learning、Distributional RL、Noisy Nets）

**优势**：
- 性能最强的 DQN 变体
- 集成多种优化技术
- 样本效率显著提升

**劣势**：
- 实现复杂度高
- 超参数调优困难
- 计算资源消耗大
- 调试困难

**适用场景**：追求最高性能，有充足计算资源

---

### 方案五：PPO (Proximal Policy Optimization)

**核心思想**：策略梯度方法，通过限制策略更新幅度保证训练稳定性。

**优势**：
- 训练稳定，不容易崩溃
- 样本效率比 DQN 高
- 可以处理连续和离散动作空间
- 目前最流行的策略梯度算法

**劣势**：
- 实现比 DQN 复杂
- 需要调整的超参数较多
- 训练时间较长

**适用场景**：需要稳定训练的场景，适合连续动作空间

---

### 方案六：A2C (Advantage Actor-Critic)

**核心思想**：结合 Actor（策略）和 Critic（价值），同时优化两者。

**优势**：
- 比纯策略梯度方法更稳定
- 可以在线学习
- 实现相对简洁
- 适合并行训练

**劣势**：
- 需要平衡 Actor 和 Critic 的学习率
- 样本效率不如 PPO
- 可能出现训练不稳定

**适用场景**：需要在线学习和快速迭代

---

### 方案七：SAC (Soft Actor-Critic)

**核心思想**：最大熵强化学习，同时优化期望奖励和策略熵。

**优势**：
- 样本效率很高
- 探索能力强
- 训练稳定
- 适合连续动作空间

**劣势**：
- 主要为连续动作设计，2048 需要适配
- 实现较复杂
- 超参数敏感

**适用场景**：需要高效探索的场景

---

## 推荐选择建议

| 优先级 | 推荐方案 | 理由 |
|--------|----------|------|
| 1 | PPO | 目前最流行、最稳定的算法，适合 2048 |
| 2 | Double DQN + Dueling DQN | 结合两者优势，性能好且实现适中 |
| 3 | Rainbow | 如果追求极致性能且有充足资源 |
| 4 | A2C | 适合快速原型开发和在线学习 |

## 安装依赖

```bash
pip install -r requirements.txt
```

## 训练

```bash
python train.py --algorithm <算法名> --config config/<配置文件>.yaml
```

## 推理

```bash
python inference.py --model <模型路径> --episodes <回合数>
```

## 支持的算法

- `dqn` - Deep Q-Network
- `double_dqn` - Double DQN
- `dueling_dqn` - Dueling DQN
- `rainbow` - Rainbow DQN
- `ppo` - Proximal Policy Optimization
- `a2c` - Advantage Actor-Critic
- `sac` - Soft Actor-Critic

## 许可证

MIT License
