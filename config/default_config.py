class Config:
    """默认配置"""
    
    # 游戏设置
    board_size = 4
    num_actions = 4
    
    # 训练设置
    algorithm = 'ppo'  # dqn, double_dqn, dueling_dqn, ppo
    num_episodes = 5000  # 增加训练回合
    max_steps_per_episode = 1000
    save_interval = 500  # 更频繁保存
    log_interval = 100  # 更频繁日志
    eval_interval = 200  # 更频繁评估
    eval_episodes = 20  # 更多评估回合
    
    # DQN 设置
    dqn = {
        'hidden_dim': 256,
        'lr': 5e-4,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.997,
        'target_update': 1,
        'buffer_size': 50000,
        'batch_size': 128,
        'prioritized_replay': True,
        'priority_alpha': 0.6,
        'priority_beta': 0.4,
        'tau': 0.01,
        'grad_clip_norm': 10.0,
        'use_one_hot': True
    }
    
    # Double DQN 设置
    double_dqn = {
        **dqn,
        'double_dqn': True,
        'dueling': False
    }
    
    # Dueling DQN 设置
    dueling_dqn = {
        **dqn,
        'double_dqn': True,
        'dueling': True
    }
    
    # PPO 设置
    ppo = {
        'hidden_dim': 128,
        'lr': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'ppo_epochs': 10,
        'batch_size': 64,
        'update_interval': 2048
    }
    
    # 路径设置
    model_save_path = 'models/checkpoints'
    log_save_path = 'logs'
