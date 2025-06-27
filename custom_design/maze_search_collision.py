import pybullet as p


def check_ball_collision(env):
    """检查机器人是否与球碰撞"""
    if env.robot_id is None or env.ball_id is None:
        return False

    # 获取机器人和球之间的接触点
    contact_points = p.getContactPoints(
        bodyA=env.robot_id,
        bodyB=env.ball_id,
        physicsClientId=env.client_id
    )

    # 如果有接触点，则表示发生了碰撞
    return len(contact_points) > 0


def check_collision(env):
    """检查是否发生碰撞"""
    if env.robot_id is not None:
        # 获取与机器人接触的物体
        contact_points = p.getContactPoints(bodyA=env.robot_id, physicsClientId=env.client_id)

        # 过滤掉与地面的接触
        for contact in contact_points:
            # 如果接触的物体不是地面且不是球，则认为是碰撞
            if contact[2] != env.maze_builder.plane_id and contact[2] != env.ball_id:
                return True

    return False
