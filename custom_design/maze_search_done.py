import math
import pybullet as p


def check_done(env, observation, info):
    """检查是否终止"""
    # 1. 找到球
    if info['found_ball']:
        return True

    # 2. 超时
    if info['timeout']:
        return True

    # 3. 倾斜过大（失去平衡）
    roll = observation[5]
    pitch = observation[6]
    env.tilt_angle = math.sqrt(roll ** 2 + pitch ** 2)  # 计算总倾斜角度
    if env.tilt_angle > env.max_tilt_angle:
        return True

    # 4. 其他终止条件（如掉出地图）
    if env.robot_id is not None:
        robot_pos, _ = p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id)
        # 检查是否掉出地图
        if robot_pos[2] < -1.0:  # z坐标小于-1表示掉出地图
            return True

    return False
