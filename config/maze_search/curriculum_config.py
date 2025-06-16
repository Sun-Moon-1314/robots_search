# -*- coding: utf-8 -*-
"""
@File    : curriculum_config.py
@Author  : zhangjian
@Desc    : 课程学习配置
"""

# 默认课程学习配置
DEFAULT_CURRICULUM = {
    "algorithm": "SAC",
    "total_phases": 3,
    "timesteps_per_phase": 500000,
    "eval_freq": 10000,
    "phases": {
        1: {
            "name": "平衡学习",
            "env_config": {
                "maze_size": (7, 7),
                "curriculum_phase": 1,
                "ball_pos": (1, 2),
                "goal_pos": "ball_pos",
            },
            "model_params": {
                "SAC": {
                    "learning_rate": 1e-3,
                    "buffer_size": 100000,
                    "batch_size": 256,
                    "ent_coef": "auto",
                    "learning_starts": 1000,
                },
                "PPO": {
                    "learning_rate": 1e-3,
                    "n_steps": 2048,
                    "batch_size": 64,
                    "n_epochs": 10,
                    "ent_coef": 0.01
                }
            },
            "reward_threshold": 0.2,
            "difficulty_factor": 0.8,  # 简单阶段，训练步数减少
            "max_attempts": 2,
            "load_from_phase": None
        },
        2: {
            "name": "短距离导航",
            "env_config": {
                "maze_size": (7, 7),
                "curriculum_phase": 2,
                "ball_pos": (1, 3),
                "goal_pos": (1, 3),
            },
            "model_params": {
                "SAC": {
                    "learning_rate": 1e-4,
                    "buffer_size": 150000,
                    "batch_size": 256,
                    "ent_coef": "auto",
                    "learning_starts": 1000
                },
                "PPO": {
                    "learning_rate": 1e-4,
                    "n_steps": 2048,
                    "batch_size": 64,
                    "n_epochs": 10,
                    "ent_coef": 0.005
                }
            },
            "reward_threshold": 100.0,
            "difficulty_factor": 1.0,  # 标准难度
            "max_attempts": 3,
            "load_from_phase": 1
        },
        3: {
            "name": "长距离导航",
            "env_config": {
                "maze_size": (7, 7),
                "curriculum_phase": 3,
                "ball_pos": (1, 5),
                "goal_pos": (1, 5),
            },
            "model_params": {
                "SAC": {
                    "learning_rate": 1e-4,
                    "buffer_size": 500000,
                    "batch_size": 256,
                    "ent_coef": "auto",
                    "learning_starts": 1000
                },
                "PPO": {
                    "learning_rate": 1e-4,
                    "n_steps": 2048,
                    "batch_size": 64,
                    "n_epochs": 10,
                    "ent_coef": 0.005
                }
            },
            "reward_threshold": None,
            "difficulty_factor": 1.5,  # 困难阶段，增加训练步数
            "max_attempts": 4,
            "step_increase_factor": 1.8,  # 每次尝试增加更多步数
            "load_from_phase": 2
        }
    }
}


def create_configurable_curriculum(curriculum_config=None):
    """
    创建可配置的课程学习框架

    Args:
        curriculum_config: 自定义课程学习配置字典

    Returns:
        dict: 配置好的课程学习配置
    """
    # 使用默认配置
    config = DEFAULT_CURRICULUM.copy()

    # 如果提供了自定义配置，则更新默认配置
    if curriculum_config:
        # 递归更新配置
        def update_config(default, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in default and isinstance(default[key], dict):
                    update_config(default[key], value)
                else:
                    default[key] = value

        update_config(config, curriculum_config)

    return config
