# -*- coding: utf-8 -*-
"""
@File    : reward_functions.py
@Author  : zhangjian
@Desc    : 奖励函数实现
"""

import numpy as np
import math

from custom_design.custom_common import get_config, normalize_angle

import threading
import sys

# 创建一个全局锁，用于保护终端输出
print_lock = threading.Lock()


def update_single_line(env, distance_to_ball, tilt_angle, forward_velocity, angle_diff_deg, normalized_components,
                       reward, en, w, total_weight):
    # 在训练循环中输出日志
    debug_str = (f"阶段{env.curriculum_phase} 训练 "
                 f"步数:{env.current_step:3d} "
                 f"距离:{distance_to_ball:4.3f}m "
                 f"倾斜:{tilt_angle:4.3f}rad "
                 f"前向速度:{forward_velocity:7.4f}m/s "
                 f"角度差:{angle_diff_deg:6.2f}°")
    debug_str += " ||=======|| 奖励:"
    for key in ['distance', 'balance_fallen', 'velocity', 'direction_exploration']:
        explain = {
            'distance': "距离",
            'balance_fallen': "平衡",
            'velocity': "速度",
            'direction_exploration': "方向探索"
        }
        if en.get(key, False) and key in normalized_components:
            component_reward = normalized_components[key] * (w[key] / total_weight if total_weight > 0 else 0)
            debug_str += f" {explain[key]}:{component_reward:8.4f}"

    debug_str += f" | 总计:{reward:8.4f}     "

    # 使用锁确保输出不被其他线程打断
    with print_lock:
        # 使用 \r 覆盖同一行输出
        sys.stdout.write("\r" + debug_str)
        sys.stdout.flush()


# 在训练循环中输出日志
def update_two_lines(env, distance_to_ball, tilt_angle, forward_velocity, angle_diff_deg, normalized_components, reward,
                     en, w, total_weight):
    # 第一行：阶段信息
    line1 = (f"————阶段{env.curriculum_phase} 训练 "
             f"步数:{env.current_step:3d} "
             f"距离:{distance_to_ball:6.3f}m "
             f"倾斜:{tilt_angle:7.4f}rad "
             f"前向速度:{forward_velocity:7.4f}m/s "
             f"角度差:{angle_diff_deg:7.4f}°————")

    # 第二行：奖励信息
    line2 = "=====奖励:"
    for key in ['distance', 'balance_fallen', 'velocity', 'direction_exploration']:
        if en.get(key, False) and key in normalized_components:
            component_reward = normalized_components[key] * (w[key] / total_weight if total_weight > 0 else 0)
            line2 += f" {key}:{component_reward:8.4f}"
    line2 += f" | 总计:{reward:8.4f}====="

    # 检查是否已初始化输出区域
    if not hasattr(update_two_lines, 'initialized'):
        # 首次输出：添加空行分隔并打印初始两行
        sys.stdout.write("\n")  # 与上方日志分隔
        sys.stdout.write(line1 + "\n")
        sys.stdout.write(line2 + "\n")
        update_two_lines.initialized = True
    else:
        # 后续更新：回到两行之前，更新内容
        sys.stdout.write("\r\033[2A\033[K" + line1 + "\n" + "\r\033[K" + line2 + "\n")
    sys.stdout.flush()


def compute_reward(env, observation, info):
    """
    计算标准化的奖励函数，所有分量都缩放到[-1, 1]范围
    """
    # 观察空间索引定义 - 28维观察空间
    OBS_IDX = {
        'robot_pos': slice(0, 2),  # [0, 1] - 机器人位置 (x, y)
        'robot_yaw': 2,  # [2] - 机器人朝向角 (yaw)
        'ball_rel_pos': slice(3, 5),  # [3, 4] - 球的相对位置 (dx, dy)
        'roll': 5,  # [5] - roll角
        'pitch': 6,  # [6] - pitch角
        'roll_rate': 7,  # [7] - roll角速度
        'pitch_rate': 8,  # [8] - pitch角速度
        'laser_data': slice(9, 17),  # [9-16] - 8个激光数据
        'laser_target': slice(17, 25),  # [17-24] - 8个激光检测目标类型
        'forward_velocity': 25,  # [25] - 前进速度
        'lateral_velocity': 26,  # [26] - 侧向速度
        'movement_alignment': 27  # [27] - 移动一致性
    }
    # 获取当前课程阶段
    current_phase = getattr(env, 'curriculum_phase', 1)
    is_eval_mode = getattr(env, 'eval_mode', False)

    # 加载配置文件并获取配置
    config = get_config(current_phase)

    # 从配置中提取参数
    w = config['weights']
    rpf = config['reward_penalty_factor']
    en = config['enable']
    t = config['thresholds']

    # 初始化奖励组件
    reward_components = {}
    normalized_components = {}

    # ===== 1. 提取观察空间关键信息 =====
    # 机器人位置和朝向
    robot_pos = observation[OBS_IDX['robot_pos']]
    robot_yaw = observation[OBS_IDX['robot_yaw']]

    # 球的相对位置
    ball_rel_pos = observation[OBS_IDX['ball_rel_pos']]
    distance_to_ball = math.sqrt(ball_rel_pos[0] ** 2 + ball_rel_pos[1] ** 2)
    relative_angle = math.atan2(ball_rel_pos[1], ball_rel_pos[0])

    # 平衡状态
    roll = observation[OBS_IDX['roll']]
    pitch = observation[OBS_IDX['pitch']]
    roll_rate = observation[OBS_IDX['roll_rate']]
    pitch_rate = observation[OBS_IDX['pitch_rate']]
    tilt_angle = math.sqrt(roll ** 2 + pitch ** 2)

    # 激光数据
    laser_data = observation[OBS_IDX['laser_data']]
    laser_targets = observation[OBS_IDX['laser_target']]

    # 速度信息
    forward_velocity = observation[OBS_IDX['forward_velocity']]
    lateral_velocity = observation[OBS_IDX['lateral_velocity']]
    movement_alignment = observation[OBS_IDX['movement_alignment']]

    # ===== 2. 核心奖励计算 =====

    # 2.1 距离奖励 - 使用距离变化
    if en.get('distance', True):
        if not hasattr(env, 'prev_distance_to_ball'):
            env.prev_distance_to_ball = distance_to_ball

        # 计算距离变化
        distance_change = env.prev_distance_to_ball - distance_to_ball

        # 在奖励计算代码中调用
        max_expected_change = env.get_max_expected_change()  # 动态获取
        # 标准化距离变化到[-1, 1]范围
        if distance_to_ball <= t['close']:
            raw_distance_reward = 1.0  # 找到球
        else:
            norm_change = np.clip(distance_change / max_expected_change, -1.0, 1.0)
            raw_distance_reward = norm_change * (rpf if norm_change > 0 else (1 - rpf))
            # if env.verbose:
            #     print(
            #         f"distance_change: {distance_change:.4f}, max_expected_change: {max_expected_change:.4f}, "
            #         f"norm_change: {norm_change:.4f}")

        # 只在非评估模式下更新距离记录
        if not is_eval_mode:
            env.prev_distance_to_ball = distance_to_ball

        normalized_components['distance'] = raw_distance_reward

    if en.get('balance_fallen', True):
        fallen_threshold = t.get('fallen', 0.4)  # 摔倒阈值
        warning_threshold = t.get('warning', 0.6) * fallen_threshold  # 警告阈值，假设为摔倒阈值的60%
        max_rate = t.get('max_rate', 5.0)  # 最大角速度参考值

        # 初始化惩罚
        balance_penalty = 0.0

        # 倾斜角度奖励/惩罚：分阶段处理
        if tilt_angle < warning_threshold:
            # 安全范围内，给予小正奖励
            tilt_penalty = 0.1 * (1 - tilt_angle / warning_threshold) * (1 - rpf)  # 角度越小，奖励越大
        elif tilt_angle < fallen_threshold:
            # 警告范围内，惩罚随角度线性增加
            normalized_tilt = (tilt_angle - warning_threshold) / (fallen_threshold - warning_threshold)
            tilt_penalty = -normalized_tilt * 0.3 * (1 - rpf)  # 惩罚力度较轻
        else:
            # 摔倒范围内，惩罚显著增加
            normalized_tilt = min(tilt_angle / (fallen_threshold * 1.5), 1.0)
            tilt_penalty = - (0.3 + 0.7 * normalized_tilt) * (1 - rpf)  # 基础惩罚+随角度增加的额外惩罚
        balance_penalty += tilt_penalty

        # 角速度惩罚：只有超过一定阈值时才惩罚，但安全范围内也可以有微小奖励
        rate_threshold = max_rate * 0.4  # 角速度阈值，假设为最大角速度的40%
        normalized_roll_rate = max(abs(roll_rate) - rate_threshold, 0.0) / (max_rate - rate_threshold)
        normalized_pitch_rate = max(abs(pitch_rate) - rate_threshold, 0.0) / (max_rate - rate_threshold)
        if normalized_roll_rate > 0 or normalized_pitch_rate > 0:
            rate_penalty = -0.2 * (normalized_roll_rate + normalized_pitch_rate) * (1 - rpf)  # 角速度惩罚
        else:
            rate_penalty = 0.05 * (1 - (abs(roll_rate) + abs(pitch_rate)) / (2 * rate_threshold)) * (1 - rpf)  # 小正奖励

        balance_penalty += rate_penalty

        normalized_components['balance_fallen'] = balance_penalty

    # 2.3 速度奖励 - 针对差速驱动机器人优化，并奖励移动一致性
    if en.get('velocity', True):
        # 获取最大速度参考值，从环境参数中获取
        max_speed = getattr(env, 'move_speed', 30.0)  # 使用环境中的 move_speed，默认为 30.0

        # 考虑动作缩放的影响，_apply_action 中 forward 被缩放到 0.6
        scaled_max_speed = max_speed * 0.6  # 调整最大速度以匹配实际动作缩放

        # 标准化速度，考虑缩放后的最大速度
        normalized_velocity = np.clip(forward_velocity / scaled_max_speed, -1.0, 1.0)
        normalized_lateral = np.clip(lateral_velocity / (scaled_max_speed * 0.5), -1.0, 1.0)

        # 计算速度奖励 - 单纯奖励前进速度，不考虑角度差
        velocity_reward = normalized_velocity * rpf  # 奖励前进速度，正向移动有正奖励，后退有负奖励
        lateral_penalty = -abs(normalized_lateral) * (1 - rpf) * 0.3  # 减少侧向移动的惩罚，降低力度

        # 考虑运动一致性 - 直接奖励移动一致性
        alignment_reward = max(0.0, movement_alignment) * 0.4 * rpf  # 确保奖励为正

        # 总速度奖励
        total_velocity_reward = velocity_reward + lateral_penalty + alignment_reward

        normalized_components['velocity'] = total_velocity_reward

    # ===== 3. 条件奖励计算 =====

    # 3.1 接近目标时的倾斜惩罚
    if en.get('tilt_near_target', True):
        near_target_threshold = t.get('close', 0.5)
        if distance_to_ball <= near_target_threshold:
            proximity_factor = (near_target_threshold - distance_to_ball) / near_target_threshold * 2.0
            tilt_threshold = t.get('tilt_threshold', 0.2)

            if tilt_angle > tilt_threshold:
                normalized_tilt = min(tilt_angle / t['fallen'], 1.0)
                raw_tilt_penalty = -normalized_tilt * proximity_factor * (1 - rpf)
                normalized_components['tilt_near_target'] = raw_tilt_penalty
            else:
                normalized_components['tilt_near_target'] = 0.0
        else:
            normalized_components['tilt_near_target'] = 0.0

    # 3.2 方向探索奖励
    angle_diff = 0
    if en.get('direction_exploration', True):
        angle_diff = abs(normalize_angle(relative_angle - robot_yaw))
        angle_threshold = t.get('angle_threshold', 0.5)

        raw_direction_reward = 0.0
        if angle_diff > angle_threshold:
            if hasattr(env, 'prev_angle_diff') and env.prev_angle_diff is not None:
                angle_change = env.prev_angle_diff - angle_diff
                max_reward = t.get('max_angle_reward', 0.5)
                factor = t.get('angle_reward_factor', 5.0)
                scale = t.get('direction_exploration_scale', 1.0)
                if angle_change > 0:  # 角度差减小
                    raw_direction_reward = scale * rpf * min(angle_change * factor, max_reward)
                else:  # 角度差增大
                    raw_direction_reward = scale * (-rpf) * min(abs(angle_change) * factor, max_reward) * 0.3  # 小幅负奖励
        else:
            raw_direction_reward = rpf * 0.2  # 角度差很小时给予小额持续奖励

        normalized_components['direction_exploration'] = raw_direction_reward
        env.prev_angle_diff = angle_diff

    # ===== 4. 可选奖励计算 =====

    # 4.1 目标检测奖励
    if en.get('target_detection', True):
        laser_target_types = observation[17:25]
        ball_detected = 2 in laser_target_types

        if ball_detected:
            target_detection_reward = rpf * 0.5
            if hasattr(env, 'prev_observation') and env.prev_observation is not None:
                prev_laser_target_types = env.prev_observation[17:25]
                prev_ball_detected = 2 in prev_laser_target_types
                if not prev_ball_detected:
                    target_detection_reward += rpf * 0.5
            else:
                target_detection_reward += rpf * 0.5
        else:
            target_detection_reward = 0.0

        normalized_components['target_detection'] = target_detection_reward
        env.prev_observation = observation.copy()

    # 4.2 目标跟踪奖励
    if en.get('target_tracking', True):
        ball_indices = [i for i, target in enumerate(laser_targets) if target == 2]
        if ball_indices:
            center_index = 3.5
            avg_deviation = sum(abs(idx - center_index) for idx in ball_indices) / len(ball_indices)
            normalized_deviation = avg_deviation / 3.5
            tracking_reward = rpf * (1.0 - normalized_deviation)
            normalized_components['target_tracking'] = tracking_reward
        else:
            normalized_components['target_tracking'] = 0.0

    # 4.3 步数惩罚
    if en.get('step_penalty', False):
        current_step = getattr(env, 'current_step', 0)
        max_steps = getattr(env, 'max_steps', 1000)  # 获取环境中的最大步数限制，默认为 1000
        min_step_threshold = int(max_steps * 0.3)  # 最小步数临界值，假设为最大步数的 30%，可根据任务调整
        close_threshold = t.get('close', 0.5)  # 接近目标的距离阈值，从配置中获取

        # 根据距离是否接近目标调整惩罚策略
        if distance_to_ball <= close_threshold:
            # 接近目标时，步数惩罚力度增加，激励快速完成任务
            if current_step < min_step_threshold:
                step_penalty = -0.008 * (1 + current_step / 150)  # 接近目标时惩罚略高，但仍较轻
            else:
                excess_steps = current_step - min_step_threshold
                penalty_factor = 1 + excess_steps / (max_steps - min_step_threshold) * 3  # 惩罚因子更强，最大 4 倍
                step_penalty = -0.015 * (1 + current_step / 80) * penalty_factor  # 基础惩罚和增长率都更高
        else:
            # 远离目标时，步数惩罚力度较轻，允许更多探索
            distance_factor = min(distance_to_ball / 1.0, 1.0) ** 0.5  # 距离因子影响较小
            if current_step < min_step_threshold:
                step_penalty = -0.003 * (1 + current_step / 300) * distance_factor  # 惩罚非常轻
            else:
                excess_steps = current_step - min_step_threshold
                penalty_factor = 1 + excess_steps / (max_steps - min_step_threshold) * 1.5  # 惩罚因子增长较缓
                step_penalty = -0.007 * (1 + current_step / 150) * distance_factor * penalty_factor  # 惩罚力度较低

        normalized_components['step_penalty'] = step_penalty * (1 - rpf)
    else:
        normalized_components['step_penalty'] = 0.0

    # ===== 5. 计算总奖励 =====

    # 计算启用的权重总和
    total_weight = sum(w[key] for key, enabled in en.items() if enabled and key in normalized_components)

    # 计算加权总奖励
    reward = 0.0
    if total_weight > 0:
        for key, enabled in en.items():
            if enabled and key in normalized_components:
                normalized_weight = w[key] / total_weight
                component_reward = normalized_weight * normalized_components[key]
                reward += component_reward
                reward_components[key] = component_reward

    # 确保奖励在[-1, 1]范围内
    reward = np.clip(reward, -1.0, 1.0)

    # 调试信息
    if hasattr(env, 'verbose') and env.verbose > 0 and env.get_done():
        debug_str = f"阶段{current_phase} {'评估' if is_eval_mode else '训练'} 步数:{env.current_step}"
        debug_str += f" 距离:{distance_to_ball:.2f}m 倾斜:{tilt_angle:.2f}rad 前向速度:{forward_velocity:.2f}m/s"

        # 添加角度差信息
        angle_diff_deg = math.degrees(angle_diff) if 'angle_diff' in locals() else 0
        debug_str += f" 角度差:{angle_diff_deg:.2f}°"

        # 在训练循环中调用
        update_single_line(env, distance_to_ball, tilt_angle, forward_velocity, angle_diff_deg, normalized_components,
                           reward, en, w, total_weight)
        if env.get_done:
            print()

    return reward
