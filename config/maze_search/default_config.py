# -*- coding: utf-8 -*-
"""
@File    : default_config.py
@Author  : zhangjian
@Desc    : 默认配置参数
"""

import os
# 在任何模块中
from path_config import PROJECT_ROOT

# 路径配置
MODELS_DIR = os.path.join(PROJECT_ROOT, 'checkpoints', 'maze_search', 'sb3_models')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'checkpoints', 'maze_search', 'sb3_logs')

# 创建目录（如果不存在）
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# 环境配置
ENV_CONFIG = {
    "maze_size": (7, 7),
    "max_steps": 1000,
    "render_mode": None,
    "verbose": False
}

# 训练配置
TRAIN_CONFIG = {
    "algorithm": "SAC",
    "total_timesteps": 300000,
    "save_freq": 10000,
    "seed": 42
}

# 评估配置
EVAL_CONFIG = {
    "episodes": 5,
    "render": True
}

# 渲染模式
RENDER_MODES = ["human", "rgb_array"]
