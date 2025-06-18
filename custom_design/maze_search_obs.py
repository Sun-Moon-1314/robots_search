import numpy as np
import pybullet as p
import math
from gymnasium import spaces


def define_observation_space():
    """
    define a observation apace
    :return:
    """
    observation_space = spaces.Box(
        low=np.array([
            # 机器人位置 (x, y)
            -10.0, -10.0,
            # 机器人朝向角 (yaw)
            -np.pi,
            # 球位置 (x, y)
            -10.0, -10.0,
            # 平衡状态 (roll, pitch, roll_rate, pitch_rate)
            -np.pi / 2, -np.pi / 2, -10.0, -10.0,
            # 8个激光数据
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            # 到球距离
            0.0,
            # 朝向一致性
            -1.0,
            # 前进速度
            -10.0,
            # 侧向速度
            0.0,
            # 移动一致性
            -1.0
        ], dtype=np.float32),
        high=np.array([
            # 机器人位置 (x, y)
            10.0, 10.0,
            # 机器人朝向角 (yaw)
            np.pi,
            # 球位置 (x, y)
            10.0, 10.0,
            # 平衡状态 (roll, pitch, roll_rate, pitch_rate)
            np.pi / 2, np.pi / 2, 10.0, 10.0,
            # 8个激光数据
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
            # 到球距离
            20.0,
            # 朝向一致性
            1.0,
            # 前进速度
            10.0,
            # 侧向速度
            10.0,
            # 移动一致性
            1.0
        ], dtype=np.float32),
        dtype=np.float32
    )
    return observation_space


def _get_laser_scan(env):
    """获取8个方向的激光扫描数据"""
    # 默认返回8个方向的最大距离
    default_scan = [5.0] * 8

    if env.robot_id is None:
        return default_scan

    try:
        robot_pos, robot_orn = p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id)
        robot_x, robot_y, _ = robot_pos
        yaw = p.getEulerFromQuaternion(robot_orn)[2]

        scan_data = []
        for i in range(8):  # 8个方向
            angle = yaw + i * math.pi / 4
            dx = math.cos(angle)
            dy = math.sin(angle)

            # 执行射线检测
            ray_start = [robot_x, robot_y, 0.2]
            ray_end = [robot_x + dx * 10, robot_y + dy * 10, 0.2]  # 最大检测距离10米
            results = p.rayTest(ray_start, ray_end, physicsClientId=env.client_id)

            # 获取距离
            hit_fraction = results[0][2]
            distance = hit_fraction * 10  # 转换为实际距离
            scan_data.append(min(distance, 5.0))  # 限制最大距离为5米

        # 确保返回8个值
        if len(scan_data) != 8:
            print(f"警告: 激光扫描数据维度不正确: {len(scan_data)}，使用默认值")
            return default_scan

        return scan_data

    except Exception as e:
        print(f"获取激光扫描数据时出错: {e}")
        return default_scan


def get_observation(env):
    """获取当前观察状态，包含平衡状态信息和关键派生特征"""
    # 初始化默认观察值
    default_obs = np.zeros(22, dtype=np.float32)  # 扩展到22维

    if env.robot_id is None:
        return default_obs

    try:
        # 获取机器人位置和朝向
        robot_pos, robot_orn = p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id)
        robot_x, robot_y, _ = robot_pos
        roll, pitch, yaw = p.getEulerFromQuaternion(robot_orn)

        # 获取线性速度和角速度
        linear_vel, angular_vel = p.getBaseVelocity(env.robot_id, physicsClientId=env.client_id)
        roll_rate, pitch_rate = angular_vel[0], angular_vel[1]

        # 获取球的位置
        if env.ball_id is not None:
            ball_pos, _ = p.getBasePositionAndOrientation(env.ball_id, physicsClientId=env.client_id)
            ball_x, ball_y, _ = ball_pos

            # 计算到球的距离
            distance_to_ball = math.sqrt(
                (robot_pos[0] - ball_pos[0]) ** 2 +
                (robot_pos[1] - ball_pos[1]) ** 2
            )

            # 计算朝向与目标方向的一致性
            target_dx = ball_pos[0] - robot_pos[0]
            target_dy = ball_pos[1] - robot_pos[1]
            target_direction = math.atan2(target_dy, target_dx)
            orientation_diff = (target_direction - yaw + math.pi) % (2 * math.pi) - math.pi
            orientation_alignment = math.cos(orientation_diff)
        else:
            ball_x, ball_y = 0, 0
            distance_to_ball = float('inf')
            orientation_alignment = 0.0

        # 计算前向向量（机器人朝向的单位向量）
        rot_matrix = p.getMatrixFromQuaternion(robot_orn)
        forward_x, forward_y = rot_matrix[0], rot_matrix[3]

        # 归一化前向向量
        forward_norm = math.sqrt(forward_x ** 2 + forward_y ** 2)
        if forward_norm > 0:
            forward_x /= forward_norm
            forward_y /= forward_norm

        # 计算前进速度和侧向速度
        vx, vy = linear_vel[0], linear_vel[1]
        velocity_magnitude = math.sqrt(vx ** 2 + vy ** 2)
        forward_velocity = vx * forward_x + vy * forward_y
        lateral_velocity = abs(vx * forward_y - vy * forward_x)

        # 计算移动方向与朝向的一致性
        if velocity_magnitude > 0.1:
            movement_dir_x = vx / velocity_magnitude
            movement_dir_y = vy / velocity_magnitude
            movement_alignment = forward_x * movement_dir_x + forward_y * movement_dir_y
        else:
            movement_alignment = 0.0

        # 获取激光扫描数据
        laser_data = _get_laser_scan(env)

        # 组合所有观察数据
        obs = np.array([
            robot_x, robot_y, yaw,  # 3个基本位置朝向
            ball_x, ball_y,  # 2个球位置
            roll, pitch, roll_rate, pitch_rate,  # 4个平衡状态
            *laser_data,  # 8个激光扫描数据
            distance_to_ball,  # 1个距离信息
            orientation_alignment,  # 1个朝向一致性
            forward_velocity,  # 1个前进速度
            lateral_velocity,  # 1个侧向速度
            movement_alignment  # 1个移动一致性
        ], dtype=np.float32)

        return obs

    except Exception as e:
        print(f"获取观察值时出错: {e}")
        return default_obs
