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
    "timesteps_per_phase": 400000,
    "eval_freq": 10000,
    "seed": None,  # 添加全局种子参数，默认为None（不固定种子）
    "phases": {
        1: {
            "name": "正常训练",
            "env_config": {
                "maze_size": (7, 7),
                "curriculum_phase": 1,
                "random_positions": [(2, 1)],
                "ball_pos": None,
                "goal_pos": None,
                "max_tilt_angle": 0.5  # 约40度，初始阶段允许较大倾斜
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
            "reward_threshold": 30,
            "difficulty_factor": 0.8,  # 简单阶段，训练步数减少
            "max_attempts": 2,
            "load_from_phase": None,
            "seed": 100,  # 阶段特定种子，如果设置则覆盖全局种子
            "performance_thresholds": {
                "max_steps": 200,  # 完成任务的最大步数阈值
                "tilt_angle": 0.4,  # 最大倾斜角度阈值（弧度），对应奖励函数中的tilt_angle
                "distance_to_target": 0.5,  # 最大距离误差阈值（米），对应奖励函数中的distance_to_ball
                "velocity_alignment": 0.0  # 移动方向与朝向的一致性阈值，对应奖励函数中的movement_alignment
            }
        },
        2: {
            "name": "短距离导航",
            "env_config": {
                "maze_size": (7, 7),
                "curriculum_phase": 2,
                "random_positions": [(2, 1)],
                "ball_pos": None,
                "goal_pos": None,
                "max_tilt_angle": 0.7  # 约40度，初始阶段允许较大倾斜
            },
            "model_params": {
                "SAC": {
                    "learning_rate": 5e-4,
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
            "reward_threshold": 30.0,
            "difficulty_factor": 1.5,  # 标准难度
            "max_attempts": 3,
            "load_from_phase": 1,
            "seed": 100,  # 阶段特定种子
            "performance_thresholds": {
                "max_steps": 200,
                "tilt_angle": 0.4,
                "distance_to_target": 0.5,
                "velocity_alignment": 0.8
            }
        },
        3: {
            "name": "长距离导航",
            "env_config": {
                "maze_size": (7, 7),
                "curriculum_phase": 3,
                "random_positions": [(1, 5), (5, 1), (4, 4), (5, 3), (3, 5)],
                "ball_pos": None,
                "goal_pos": None,
                "max_tilt_angle": 0.7  # 约40度，初始阶段允许较大倾斜
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
            "difficulty_factor": 3.0,  # 困难阶段，增加训练步数
            "max_attempts": 4,
            "step_increase_factor": 1.8,  # 每次尝试增加更多步数
            "load_from_phase": 2,
            "seed": None,  # 阶段特定种子
            "performance_thresholds": {
                "max_steps": 300,
                "tilt_angle": 0.5,
                "distance_to_target": 0.5,
                "velocity_alignment": 0.6
            }
        }
    }
}
# 正常训练
TRAINER_CONFIG = {
    "name": "轮式机器人自主搜索",
    "algorithm": "SAC",
    "timesteps_per_phase": 1000000,
    "eval_freq": 10000,
    "env_config": {
        "maze_size": (7, 7),
        "curriculum_phase": 1,
        "random_positions": [(2, 1)],
        "ball_pos": None,
        "goal_pos": None,
        "max_tilt_angle": 0.4,  # 约40度，初始阶段允许较大倾斜
        "max_expected_change": 0.2
    },
    "model_params": {
        "SAC": {
            "learning_rate": 5e-4,
            "buffer_size": 300000,
            "batch_size": 256,
            "ent_coef": "auto",
            "learning_starts": 1000,
        },
        "PPO": {
            "learning_rate": 1e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "ent_coef": 0.01
        }
    },
    "reward_threshold": None,
    "seed": 100,
    "performance_thresholds": {
        "tilt_angle": 0.4,  # 最大倾斜角度阈值（弧度），对应奖励函数中的tilt_angle
        "distance_to_target": 0.4,  # 最大距离误差阈值（米），对应奖励函数中的distance_to_ball
        "movement_alignment": 0.8  # 移动方向与朝向的一致性阈值，对应奖励函数中的movement_alignment
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
    config = TRAINER_CONFIG.copy()

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


def get_reward_config(curriculum_phase):
    """根据课程阶段返回相应的奖励参数配置"""
    # 基础配置
    config = {
        # 奖励权重 - 使用1-10的整数范围
        'weights': {
            'distance': 0,
            'balance_fallen': 0,
            'velocity': 0,
            'fallen': 0,  # 摔倒惩罚
            'tilt_near_target': 0,
            'direction_exploration': 0,
            'target_detection': 0,
            'target_tracking': 0,
            'step_penalty': 0
        },
        # 奖励-惩罚因子
        'reward_penalty_factor': 0.5,
        # 奖励开关
        'enable': {
            'distance': False,  # 默认全部关闭，根据阶段开启
            'balance_fallen': False,
            'velocity': False,
            'tilt_near_target': False,
            'direction_exploration': False,
            'target_detection': False,
            'target_tracking': False,
            'step_penalty': False
        },
        # 阈值设置
        'thresholds': {
            'close': 0.4,  # 接近目标阈值(米)：机器人与目标距离小于此值算接近。值越大，判定越宽松；值越小，要求更近。
            'fallen': 0.5,  # 摔倒阈值(弧度)：倾斜角度超此值算摔倒。值越大，越不容易摔倒；值越小，越容易摔倒。
            'warning': 0.3,  # 摔倒预警阈值(弧度)：接近摔倒时的警告角度。值越大，预警越早；值越小，预警越晚。
            'tilt_threshold': 0.1,  # 倾斜阈值(弧度)：特定场景倾斜惩罚阈值。值越大，容忍度越高；值越小，容忍度越低。
            'angle_threshold': 0.1,  # 侧面目标角度阈值(弧度)：目标偏离朝向的判定角度。值越大，范围越宽；值越小，要求更正对。
            'max_angle_reward': 0.6,  # 最大角度奖励：方向调整的最大奖励值。值越大，奖励上限越高；值越小，奖励上限越低。
            'angle_reward_factor': 3.0,  # 角度变化奖励因子：放大角度变化的奖励影响。值越大，影响越强；值越小，影响越弱。
            'direction_exploration_scale': 1.0,  # 方向探索缩放系数：调整方向探索奖励幅度。值越大，奖励越强；值越小，奖励越弱。
            'max_rate': 5.0  # 角速度最大值(弧度/秒)：归一化角速度的参考值。值越大，敏感度越低；值越小，敏感度越高。

        }
    }

    # 根据课程阶段设置具体参数
    if curriculum_phase == 1:  # 阶段0：普通训练模式（非课程学习）
        # 开启所有奖励组件
        config['enable'].update({
            'distance': True,  # 距离目标
            'balance_fallen': True,  # 平衡
            'velocity': True,  # 速度
            'direction_exploration': True,  # 方向探索
            'target_detection': True,  # 目标检测
            'target_tracking': True,  # 目标跟踪
            'tilt_near_target': True,  # 接近目标时保持平衡
            'step_penalty': True  # 步数效率
        })

        # 设置权重 (1-10范围)，均衡分配，稍微偏向核心目标
        config['weights'].update({
            'distance': 7,  # 距离目标较重要
            'balance_fallen': 5,  # 平衡重要
            'fallen': 4,  # 摔倒惩罚重要
            'velocity': 2,  # 速度奖励
            'direction_exploration': 3,  # 方向探索
            'target_detection': 2,  # 目标检测
            'target_tracking': 1,  # 目标跟踪
            'tilt_near_target': 5,  # 接近目标时的平衡
            'step_penalty': 2  # 步数惩罚
        })
        config['reward_penalty_factor'] = 0.6  # 平衡奖励和惩罚

    return config
