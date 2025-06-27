import numpy as np
import pybullet as p
import math
from gymnasium import spaces


def define_observation_space():
    """
    定义观察空间，使用相对位置替代球的绝对位置和方向一致性，并添加激光雷达检测小球的信息
    """
    # 28维度
    observation_space = spaces.Box(
        low=np.array([
            # 机器人位置 (x, y)
            -10.0, -10.0,
            # 机器人朝向角 (yaw)
            -np.pi,
            # 球相对位置 (dx, dy)
            -20.0, -20.0,
            # 平衡状态 (roll, pitch, roll_rate, pitch_rate)
            -np.pi / 2, -np.pi / 2, -10.0, -10.0,
            # 8个激光数据
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            # 8个激光检测目标类型 (0=无目标, 1=墙壁, 2=小球)
            0, 0, 0, 0, 0, 0, 0, 0,
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
            # 球相对位置 (dx, dy)
            20.0, 20.0,
            # 平衡状态 (roll, pitch, roll_rate, pitch_rate)
            np.pi / 2, np.pi / 2, 10.0, 10.0,
            # 8个激光数据
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
            # 8个激光检测目标类型 (0=无目标, 1=墙壁, 2=小球)
            2, 2, 2, 2, 2, 2, 2, 2,
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


def _get_laser_scan_with_types(env):
    """获取8个方向的激光扫描数据，包括距离和目标类型"""
    # 默认返回8个方向的最大距离和目标类型
    default_scan = [(5.0, 0)] * 8  # (距离, 目标类型) - 0表示无目标, 1表示墙壁, 2表示小球

    if env.robot_id is None:
        return default_scan

    try:
        robot_pos, robot_orn = p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id)
        robot_x, robot_y, robot_z = robot_pos
        yaw = p.getEulerFromQuaternion(robot_orn)[2]

        # 激光高度和俯仰角设置
        laser_height = 0.3  # 降低激光起始高度，默认是0.2
        pitch_angle = -0.2  # 向下的俯仰角（弧度），负值表示向下

        scan_data = []
        for i in range(8):  # 8个方向
            angle = yaw + i * math.pi / 4
            dx = math.cos(angle)
            dy = math.sin(angle)
            dz = math.sin(pitch_angle)  # 添加Z轴方向的分量，实现向下倾斜

            # 计算水平方向的分量需要考虑俯仰角
            horizontal_component = math.cos(pitch_angle)
            dx *= horizontal_component
            dy *= horizontal_component

            # 执行射线检测
            ray_start = [robot_x, robot_y, laser_height]  # 降低激光起始高度
            ray_end = [robot_x + dx * 10, robot_y + dy * 10, laser_height + dz * 10]  # 最大检测距离10米，考虑Z轴偏移
            results = p.rayTest(ray_start, ray_end, physicsClientId=env.client_id)
            # print(f"{results}")
            # 获取距离和检测到的物体ID
            hit_fraction = results[0][2]
            hit_object_id = results[0][0]  # 射线碰到的物体ID

            # 计算实际3D距离
            dx_hit = dx * hit_fraction * 10
            dy_hit = dy * hit_fraction * 10
            dz_hit = dz * hit_fraction * 10
            distance = math.sqrt(dx_hit ** 2 + dy_hit ** 2 + dz_hit ** 2)

            if hit_object_id >= 0:  # 有效的物体ID
                if hit_object_id == env.ball_id:
                    target_type = 2  # 小球
                    # 打印调试信息，当检测到小球时
                    # print(f"检测到小球! 传感器:{i}, 距离:{distance:.2f}, 物体ID:{hit_object_id}, 球ID:{env.ball_id}")
                    scan_data.append((min(distance, 5.0), target_type))  # 限制最大距离为5米
                else:
                    target_type = 1  # 墙壁或其他物体
                    scan_data.append((min(distance, 5.0), target_type))  # 限制最大距离为5米
            else:
                # 确定目标类型
                target_type = 0  # 默认为无目标
                scan_data.append((min(distance, 5.0), target_type))  # 限制最大距离为5米

        if env.verbose > 2:
            for index, data in enumerate(scan_data):
                if data[1] == 2:
                    print(f"scan_data: 第{index}个激光传感器，检测到了小球: 距离为{data[0]:.2f}")

        # 确保返回8个值
        if len(scan_data) != 8:
            print(f"警告: 激光扫描数据维度不正确: {len(scan_data)}，使用默认值")
            return default_scan

        return scan_data

    except Exception as e:
        print(f"获取激光扫描数据时出错: {e}")
        return default_scan


def _get_laser_scan(env):
    """获取8个方向的激光扫描数据（仅距离）"""
    # 获取带有目标类型的激光扫描数据
    scan_data_with_types = _get_laser_scan_with_types(env)

    # 仅返回距离部分
    return [dist for dist, _ in scan_data_with_types]


def get_observation(env):
    """获取当前观察状态，使用相对位置替代球的绝对位置和方向一致性，并添加激光雷达检测小球的信息"""
    # 初始化默认观察值 - 现在是28维（增加了8个激光目标类型）
    default_obs = np.zeros(28, dtype=np.float32)

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

        # 计算球的相对位置
        target_dx, target_dy = 0, 0
        if env.ball_id is not None:
            ball_pos, _ = p.getBasePositionAndOrientation(env.ball_id, physicsClientId=env.client_id)
            # 计算球相对于机器人的位置（在世界坐标系中）
            target_dx = ball_pos[0] - robot_pos[0]
            target_dy = ball_pos[1] - robot_pos[1]

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

        # 获取激光扫描数据和目标类型
        laser_data_with_types = _get_laser_scan_with_types(env)
        # print(f"laser_data_with_types:{laser_data_with_types}")
        laser_distances = [data[0] for data in laser_data_with_types]
        # print(f"laser_distances:{laser_distances}")
        laser_target_types = [data[1] for data in laser_data_with_types]
        # if 2 in laser_target_types:
        #     print(f"laser_target_types:{laser_target_types}")

        # 组合所有观察数据
        obs = np.array([
            robot_x, robot_y,  # 机器人位置
            yaw,  # 机器人朝向角
            target_dx, target_dy,  # 球的相对位置
            roll, pitch, roll_rate, pitch_rate,  # 机器人4个平衡状态
            *laser_distances,  # 机器人8个激光扫描距离
            *laser_target_types,  # 机器人8个激光扫描目标类型
            forward_velocity,  # 1个前进速度
            lateral_velocity,  # 1个侧向速度
            movement_alignment  # 1个移动一致性
        ], dtype=np.float32)

        # if env.verbose > 0:
        # if 2 in obs[17:25].tolist():
        #     print("============================================================")
        #     print("----- 最终观察值 -----")
        #     # print(f"维度: {len(obs)}")
        #     # print("观察值数组:")
        #     print(f"  机器人位置: [{obs[0]:.4f}, {obs[1]:.4f}]")
        #     print(f"  机器人朝向: {obs[2]:.4f}")
        #     print(f"  球相对位置: [{obs[3]:.4f}, {obs[4]:.4f}]")
        #     print(f"  平衡状态: [{obs[5]:.4f}, {obs[6]:.4f}, {obs[7]:.4f}, {obs[8]:.4f}]")
        #     print(f"  激光距离: {obs[9:17].tolist()}")
        #     print(f"  目标类型: {obs[17:25].tolist()}")
        #     print(f"  速度与一致性: [{obs[25]:.4f}, {obs[26]:.4f}, {obs[27]:.4f}]")
        #     print("============================================================")

        return obs

    except Exception as e:
        print(f"获取观察值时出错: {e}")
        return default_obs
