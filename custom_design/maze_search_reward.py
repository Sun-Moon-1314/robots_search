# -*- coding: utf-8 -*-
"""
@File    : reward_functions.py
@Author  : zhangjian
@Desc    : 奖励函数实现
"""

import numpy as np
import math


def get_reward_config(curriculum_phase):
    """
    根据课程阶段返回相应的奖励参数配置

    Args:
        curriculum_phase (int): 课程学习阶段 (1, 2, 3)

    Returns:
        dict: 包含该阶段所有奖励参数的字典
    """
    # 基础配置 - 所有阶段共享的参数结构
    config = {
        # 奖励权重 - 统一管理所有权重
        'weights': {
            # 核心奖励
            'distance': 0.0,  # 与目标距离相关的奖励权重
            'balance': 0.0,  # 保持平衡的奖励权重
            'fallen': 0.0,  # 摔倒惩罚的权重
            'velocity': 0.0,  # 速度相关奖励的权重

            # 可选奖励
            'collision': 0.0,  # 碰撞惩罚的权重
            'energy': 0.0,  # 能量消耗惩罚的权重
            'alive': 0.0,  # 存活奖励的权重
            'orientation_penalty': 0.0,  # 朝向目标的奖励权重
            'progress': 0.0,  # 任务进度奖励权重
            'lateral': 0.0,  # 侧向移动惩罚的权重
            'alignment': 0.0,  # 移动方向与朝向一致性奖励的权重
        },

        # 奖励开关 - 控制哪些奖励被启用
        'enable': {
            # 核心奖励总是启用
            'distance': True,
            'balance': True,
            'fallen': True,
            'velocity': True,

            # 可选奖励默认禁用
            'collision': False,
            'energy': False,
            'alive': False,
            'orientation_penalty': False,
            'progress': False,
            'lateral': False,
            'alignment': False,
        },

        # 阈值设置
        'thresholds': {
            'close': 0.0,  # 接近目标的阈值
            'fallen': 0.0,  # 摔倒判定阈值
            'warning': 0.0,  # 摔倒预警阈值
        },

        # 奖励缩放因子
        'scales': {
            'distance_change': 0.0,  # 距离变化的缩放因子
            'velocity': 0.0,  # 速度奖励的缩放因子
        },

        # 奖励上下限
        'limits': {
            'distance_max': 0.0,  # 距离奖励上限
            'distance_min': 0.0,  # 距离奖励下限
        },

        # 固定奖励值
        'values': {
            'goal_reached': 0.0,  # 到达目标的奖励
            'fallen_penalty': 0.0,  # 摔倒的惩罚
        }
    }

    # 根据课程阶段设置具体参数
    if curriculum_phase == 1:  # 阶段1：学习平衡
        # 设置权重
        config['weights'].update({
            # 核心奖励
            'distance': 0.28,  # 中等重要性
            'balance': 0.30,  # 高重要性
            'fallen': 0.22,  # 高重要性
            'velocity': 0.05,  # 低-中等重要性

            # 可选奖励
            'energy': 0.05,  # 低重要性
            'orientation_penalty': 0.10,  # 低重要性

            # 其他可选奖励权重为0
            'collision': 0.00,
            'alive': 0.00,
            'progress': 0.00,
            'lateral': 0.00,
            'alignment': 0.00,
        })

        # 启用部分可选奖励
        config['enable'].update({
            'energy': True,  # 启用能量消耗惩罚
            'orientation_penalty': True,  # 启用朝向奖励
        })

        config['thresholds'].update({
            'close': 0.3,
            'fallen': 0.6,
            'warning': 0.4,
        })

        config['scales'].update({
            'distance_change': 1.0,
            'velocity': 1.5,
        })

        config['limits'].update({
            'distance_max': 2.0,
            'distance_min': -1.0,
        })

        config['values'].update({
            'goal_reached': 1.0,  # 标准化为1
            'fallen_penalty': -1.0,  # 标准化为-1
        })

    elif curriculum_phase == 2:  # 阶段2：学习短距离导航
        # 设置权重
        config['weights'].update({
            # 核心奖励
            'distance': 0.20,  # 高重要性
            'balance': 0.30,  # 中-高重要性
            'fallen': 0.15,  # 中-高重要性
            'velocity': 0.05,  # 中等重要性

            # 可选奖励
            'energy': 0.05,  # 低-中等重要性
            'lateral': 0.05,  # 中等重要性
            'alignment': 0.20,  # 中等重要性

            # 其他可选奖励权重为0
            'collision': 0.00,
            'alive': 0.00,
            'orientation_penalty': 0.00,  # 不使用，改用alignment
            'progress': 0.00,
        })

        # 启用部分可选奖励
        config['enable'].update({
            'lateral': True,  # 启用侧向移动惩罚
            'alignment': True,  # 启用方向一致性奖励
            'energy': True,  # 启用能量消耗惩罚
            'orientation_penalty': False,  # 禁用朝向奖励，使用alignment代替
        })

        config['thresholds'].update({
            'close': 0.4,
            'fallen': 0.65,
            'warning': 0.45,
        })

        config['scales'].update({
            'distance_change': 3.0,
            'velocity': 2.0,
        })

        config['limits'].update({
            'distance_max': 2.5,
            'distance_min': -1.5,
        })

        config['values'].update({
            'goal_reached': 1.0,  # 标准化为1
            'fallen_penalty': -1.0,  # 标准化为-1
        })

    elif curriculum_phase == 3:  # 阶段3：学习长距离导航
        # 设置权重
        config['weights'].update({
            # 核心奖励
            'distance': 0.25,  # 高重要性
            'balance': 0.15,  # 中-高重要性
            'fallen': 0.15,  # 中-高重要性
            'velocity': 0.10,  # 中等重要性

            # 可选奖励
            'collision': 0.05,  # 低-中等重要性
            'energy': 0.05,  # 低-中等重要性
            'progress': 0.05,  # 低-中等重要性
            'lateral': 0.10,  # 中-高重要性
            'alignment': 0.10,  # 中-高重要性

            # 其他可选奖励权重为0
            'alive': 0.00,
            'orientation_penalty': 0.00,  # 不使用，改用alignment
        })

        # 启用更多可选奖励
        config['enable'].update({
            'lateral': True,  # 启用侧向移动惩罚
            'alignment': True,  # 启用方向一致性奖励
            'energy': True,  # 启用能量消耗惩罚
            'collision': True,  # 启用碰撞惩罚
            'progress': True,  # 启用进度奖励
            'orientation_penalty': False,  # 禁用朝向奖励，使用alignment代替
        })

        config['thresholds'].update({
            'close': 0.5,
            'fallen': 0.7,
            'warning': 0.5,
        })

        config['scales'].update({
            'distance_change': 5.0,
            'velocity': 3.0,
        })

        config['limits'].update({
            'distance_max': 3.0,
            'distance_min': -2.0,
        })

        config['values'].update({
            'goal_reached': 1.0,  # 标准化为1
            'fallen_penalty': -1.0,  # 标准化为-1
        })

    # 验证启用的权重总和是否为1.0
    enabled_weights_sum = sum(config['weights'][key] for key, enabled in config['enable'].items() if enabled)

    if abs(enabled_weights_sum - 1.0) > 1e-6:
        print(f"警告: 阶段{curriculum_phase}的启用权重总和为{enabled_weights_sum}，不等于1.0")
        # 自动归一化启用的权重
        normalize_factor = 1.0 / enabled_weights_sum if enabled_weights_sum > 0 else 0.0
        for key, enabled in config['enable'].items():
            if enabled:
                config['weights'][key] *= normalize_factor

    return config


def compute_reward(env, observation, info):
    """
    计算标准化的奖励函数，所有分量都缩放到[-1, 1]范围

    Args:
        env: 环境实例
        observation: 观察空间中的状态信息 (22维向量)
        info: 当前状态信息

    Returns:
        float: 计算得到的奖励值
    """
    # 观察空间索引定义
    OBS_IDX = {
        'robot_pos': slice(0, 2),  # [0, 1] - 机器人位置 (x, y)
        'robot_yaw': 2,  # [2] - 机器人朝向角 (yaw)
        'ball_pos': slice(3, 5),  # [3, 4] - 球位置 (x, y)
        'roll': 5,  # [5] - roll角
        'pitch': 6,  # [6] - pitch角
        'roll_rate': 7,  # [7] - roll角速度
        'pitch_rate': 8,  # [8] - pitch角速度
        'laser_data': slice(9, 17),  # [9-16] - 8个激光数据
        'distance_to_ball': 17,  # [17] - 到球距离
        'orientation_alignment': 18,  # [18] - 朝向一致性
        'forward_velocity': 19,  # [19] - 前进速度
        'lateral_velocity': 20,  # [20] - 侧向速度
        'movement_alignment': 21  # [21] - 移动一致性
    }

    # 获取当前课程阶段
    current_phase = getattr(env, 'curriculum_phase', 3)

    # 检查是否为评估模式
    is_eval_mode = getattr(env, 'eval_mode', False)

    # 获取当前阶段的奖励配置
    config = get_reward_config(current_phase)

    # 从配置中提取参数
    w = config['weights']  # 权重
    en = config['enable']  # 奖励开关
    t = config['thresholds']  # 阈值
    v = config['values']  # 固定值

    # 初始化奖励组件
    reward_components = {}
    normalized_components = {}  # 存储标准化后的分量

    # ===== 核心奖励计算 =====

    # 1. 距离奖励计算 - 直接从观察空间获取
    distance_to_ball = observation[OBS_IDX['distance_to_ball']]

    # 确保prev_distance_to_ball属性存在
    if not hasattr(env, 'prev_distance_to_ball'):
        env.prev_distance_to_ball = None

    # 初始化距离奖励
    raw_distance_reward = 0.0

    if env.prev_distance_to_ball is not None:
        # 计算距离变化
        distance_change = env.prev_distance_to_ball - distance_to_ball

        if distance_to_ball <= t['close']:
            # 找到球 - 给予最高奖励
            raw_distance_reward = 1.0  # 已标准化为1
        else:
            # 使用距离变化的标准化奖励
            # 假设最大合理距离变化为±0.1单位/步
            max_expected_change = 0.1

            # 标准化距离变化到[-1, 1]
            norm_distance_change = np.clip(distance_change / max_expected_change, -1.0, 1.0)

            # 距离增加时惩罚更强
            if distance_change < 0:
                raw_distance_reward = 2.0 * norm_distance_change  # 仍在[-1, 0]范围内
            else:
                raw_distance_reward = norm_distance_change  # 在[0, 1]范围内

    # 只在非评估模式下更新距离记录
    if not is_eval_mode:
        env.prev_distance_to_ball = distance_to_ball

    reward_components['distance'] = raw_distance_reward
    normalized_components['distance'] = raw_distance_reward  # 已经标准化

    # 2. 平衡奖励计算 - 从观察空间获取
    roll = observation[OBS_IDX['roll']]
    pitch = observation[OBS_IDX['pitch']]
    tilt_angle = math.sqrt(roll ** 2 + pitch ** 2)  # 计算总倾斜角度

    # 标准化平衡奖励到[-1, 0]范围
    # 假设最大倾斜角度为t['fallen']
    max_tilt = t['fallen']
    normalized_tilt = min(tilt_angle / max_tilt, 1.0)  # 在[0, 1]范围内
    raw_balance_reward = -normalized_tilt ** 2  # 在[-1, 0]范围内

    reward_components['balance'] = raw_balance_reward
    normalized_components['balance'] = raw_balance_reward  # 已经标准化

    # 3. 摔倒检测和惩罚
    is_fallen = tilt_angle > t['fallen']

    # 标准化摔倒惩罚到[-1, 0]范围
    if is_fallen:
        raw_fallen_penalty = -1.0  # 最大惩罚
    else:
        # 接近摔倒阈值时给予预警惩罚
        if tilt_angle > t['warning']:
            # 在warning_threshold和fallen_threshold之间线性增加惩罚
            warning_factor = (tilt_angle - t['warning']) / (t['fallen'] - t['warning'])
            raw_fallen_penalty = -warning_factor  # 在[-1, 0]范围内
        else:
            raw_fallen_penalty = 0.0

    reward_components['fallen'] = raw_fallen_penalty
    normalized_components['fallen'] = raw_fallen_penalty  # 已经标准化

    # 4. 速度奖励计算 - 从观察空间获取
    forward_velocity = observation[OBS_IDX['forward_velocity']]

    # 标准化速度奖励到[-1, 1]范围
    # 假设最大合理速度为1.0单位/步
    max_expected_velocity = 1.0
    normalized_velocity = np.clip(forward_velocity / max_expected_velocity, -1.0, 1.0)

    if forward_velocity > 0:
        raw_velocity_reward = normalized_velocity  # 在[0, 1]范围内
    else:
        raw_velocity_reward = 2.5 * normalized_velocity  # 在[-1, 0]范围内，后退惩罚更强

    reward_components['velocity'] = raw_velocity_reward
    normalized_components['velocity'] = raw_velocity_reward  # 已经标准化

    # ===== 可选奖励计算 =====

    # 5. 朝向奖励（可选）- 从观察空间获取
    if en.get('orientation_penalty', False):
        orientation_alignment = observation[OBS_IDX['orientation_alignment']]
        # orientation_alignment值为1表示完全一致，0表示垂直，-1表示完全相反

        # 新逻辑：只要不完全一致就给予惩罚，惩罚程度与偏离程度成正比
        if orientation_alignment < 1.0:  # 只有完全一致(=1)时才不惩罚
            # 将[1, -1]映射到[0, 1]范围，1表示无偏差，-1表示完全相反
            deviation = (1.0 - orientation_alignment) / 2.0  # 归一化到[0, 1]

            # 二次惩罚：偏离越大，惩罚增长越快
            orientation_penalty = deviation ** 2

            reward_components['orientation_penalty'] = -orientation_penalty  # 负值表示惩罚
            normalized_components['orientation_penalty'] = -orientation_penalty
        else:
            # 朝向完全正确时不惩罚
            reward_components['orientation_penalty'] = 0.0
            normalized_components['orientation_penalty'] = 0.0
    else:
        reward_components['orientation_penalty'] = 0.0
        normalized_components['orientation_penalty'] = 0.0

    # 6. 侧向速度惩罚（可选）- 从观察空间获取
    if en.get('lateral', False):
        lateral_velocity = observation[OBS_IDX['lateral_velocity']]
        # 标准化侧向速度到[-1, 0]范围
        max_expected_lateral = 0.5  # 假设最大合理侧向速度
        normalized_lateral = min(abs(lateral_velocity) / max_expected_lateral, 1.0)
        raw_lateral_penalty = -normalized_lateral  # 在[-1, 0]范围内
        reward_components['lateral'] = raw_lateral_penalty
        normalized_components['lateral'] = raw_lateral_penalty
    else:
        reward_components['lateral'] = 0.0
        normalized_components['lateral'] = 0.0

    # 7. 移动方向与朝向一致性奖励（可选）- 从观察空间获取
    if en.get('alignment', False):
        movement_alignment = observation[OBS_IDX['movement_alignment']]
        # 假设movement_alignment在[-1, 1]范围内

        # 非线性惩罚 - 不对齐时惩罚更严厉
        raw_alignment_penalty = max(0, (1 - movement_alignment) ** 2)

        reward_components['alignment'] = -raw_alignment_penalty
        normalized_components['alignment'] = -raw_alignment_penalty
    else:
        reward_components['alignment'] = 0.0
        normalized_components['alignment'] = 0.0

    # 8. 能量效率奖励（可选）
    if en.get('energy', False):
        actions = info.get('actions', np.zeros(2))
        # 标准化能量惩罚到[-1, 0]范围
        # 假设最大合理能量消耗为actions的平方和为4
        max_expected_energy = 4.0
        normalized_energy = min(np.sum(np.square(actions)) / max_expected_energy, 1.0)
        raw_energy_penalty = -normalized_energy  # 在[-1, 0]范围内
        reward_components['energy'] = raw_energy_penalty
        normalized_components['energy'] = raw_energy_penalty
    else:
        reward_components['energy'] = 0.0
        normalized_components['energy'] = 0.0

    # 9. 碰撞惩罚（可选）- 仍从info获取，因为这是环境状态而非观察
    if en.get('collision', False):
        # 碰撞是二元事件，直接映射到{-1, 0}
        raw_collision_penalty = -1.0 if info['collision'] else 0.0
        reward_components['collision'] = raw_collision_penalty
        normalized_components['collision'] = raw_collision_penalty
    else:
        reward_components['collision'] = 0.0
        normalized_components['collision'] = 0.0

    # 10. 存活奖励（可选）
    if en.get('alive', False):
        # 存活是二元事件，直接映射到{0, 1}
        raw_alive_bonus = 1.0  # 最大为1
        reward_components['alive'] = raw_alive_bonus
        normalized_components['alive'] = raw_alive_bonus
    else:
        reward_components['alive'] = 0.0
        normalized_components['alive'] = 0.0

    # 11. 进度奖励（可选）- 使用观察空间中的距离信息
    if en.get('progress', False):
        # 确保initial_distance_to_ball属性存在
        if not hasattr(env, 'initial_distance_to_ball'):
            env.initial_distance_to_ball = info.get('initial_distance_to_ball', distance_to_ball)

        initial_distance = env.initial_distance_to_ball
        progress = 1.0 - (distance_to_ball / initial_distance)
        progress = max(0.0, min(1.0, progress))  # 已标准化到[0, 1]
        raw_progress_reward = progress
        reward_components['progress'] = raw_progress_reward
        normalized_components['progress'] = raw_progress_reward
    else:
        reward_components['progress'] = 0.0
        normalized_components['progress'] = 0.0

    # 计算总奖励
    reward = 0.0

    # 使用标准化后的奖励分量和权重计算总奖励
    for key, enabled in en.items():
        if enabled:
            reward += w[key] * normalized_components[key]

    # 调试信息
    if hasattr(env, 'verbose') and env.verbose >= 2 and env.current_step % 1000 == 0:
        debug_str = f"阶段{current_phase} {'评估' if is_eval_mode else '训练'} 奖励明细: "

        # 添加所有启用的奖励信息
        for key, enabled in en.items():
            if enabled:
                debug_str += f"{key}={normalized_components[key]:.2f}*{w[key]:.2f}={w[key] * normalized_components[key]:.2f}, "

        debug_str += f"总计={reward:.2f}"
        print(debug_str)

    return reward
