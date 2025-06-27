import math

import numpy as np
import pybullet as p

from custom_design.maze_search_collision import check_ball_collision, check_collision


def get_info(env):
    """获取当前环境状态的信息，仅包含不作为观察空间一部分的额外监控信息"""

    INFO = {
        # 任务相关指标
        'found_ball': False,  # 是否找到球
        'timeout': False,  # 是否超时
        'progress': 0.0,  # 任务进度 (0-1)

        # 碰撞信息
        'obstacle_collision': False,  # 是否发生碰撞
        'ball_collision': False,  # 是否与球碰撞

        # 记录当前动作
        'actions': np.zeros(2)  # 当前执行的动作
    }
    # 添加当前动作到info
    if hasattr(env, 'current_action'):
        INFO['actions'] = env.current_action

    # 计算到球的距离和任务相关信息
    if env.robot_id is not None and env.ball_id is not None:
        robot_pos, _ = p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id)
        ball_pos, _ = p.getBasePositionAndOrientation(env.ball_id, physicsClientId=env.client_id)

        # 计算到球的距离 (这个值在get_obs中已经计算过，但在INFO中也需要)
        distance_to_ball = math.sqrt(
            (robot_pos[0] - ball_pos[0]) ** 2 +
            (robot_pos[1] - ball_pos[1]) ** 2
        )
        INFO['distance_to_ball'] = distance_to_ball

        # 记录初始距离（如果尚未记录）
        if not hasattr(env, 'initial_distance_to_ball'):
            env.initial_distance_to_ball = distance_to_ball
        INFO['initial_distance_to_ball'] = env.initial_distance_to_ball

        # 检查是否与球碰撞
        ball_collision = check_ball_collision(env)
        INFO['ball_collision'] = ball_collision

        # 使用碰撞检测或距离判断是否找到球
        INFO['found_ball'] = ball_collision or (distance_to_ball < 0.50)

        # 检查与障碍物的碰撞
        INFO['obstacle_collision'] = check_collision(env)

        # 计算任务进度
        progress = 1.0 - (distance_to_ball / env.initial_distance_to_ball)
        INFO['progress'] = max(0.0, min(1.0, progress))  # 限制在0到1之间

    # 检查是否超时
    if env.current_step >= env.max_steps:
        INFO['timeout'] = True

    return INFO
