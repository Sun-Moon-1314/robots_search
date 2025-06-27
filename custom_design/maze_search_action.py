import numpy as np
import pybullet as p


def apply_action(env, action):
    """应用动作到机器人"""
    # 应用动作
    if env.robot_id is not None:
        # 对动作进行平滑处理，避免突变
        if not hasattr(env, 'prev_action'):
            env.prev_action = np.zeros_like(action)

        # 平滑系数 - 值越小平滑效果越强
        smoothing = 0.3

        # 平滑动作变化
        smoothed_action = smoothing * np.array(action) + (1 - smoothing) * env.prev_action
        env.prev_action = smoothed_action

        # 解析动作（前进速度和转向）
        forward = float(smoothed_action[0]) * 0.6  # 降低最大速度
        turn = float(smoothed_action[1]) * 0.5  # 降低最大转向
        # 保存当前动作到info中供奖励函数使用
        env.current_action = np.array([forward, turn])

        # 根据打印的关节信息，R2D2有4个轮子
        right_front_wheel = 2  # right_front_wheel_joint
        right_back_wheel = 3  # right_back_wheel_joint
        left_front_wheel = 6  # left_front_wheel_joint
        left_back_wheel = 7  # left_back_wheel_joint

        # 计算轮子速度
        right_side_speed = env.move_speed * (forward - turn)
        left_side_speed = env.move_speed * (forward + turn)

        # 控制所有四个轮子
        # 右侧两个轮子
        p.setJointMotorControl2(
            bodyUniqueId=env.robot_id,
            jointIndex=right_front_wheel,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=right_side_speed,
            force=env.move_force,
            physicsClientId=env.client_id
        )

        p.setJointMotorControl2(
            bodyUniqueId=env.robot_id,
            jointIndex=right_back_wheel,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=right_side_speed,
            force=env.move_force,
            physicsClientId=env.client_id
        )

        # 左侧两个轮子
        p.setJointMotorControl2(
            bodyUniqueId=env.robot_id,
            jointIndex=left_front_wheel,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=left_side_speed,
            force=env.move_force,
            physicsClientId=env.client_id
        )

        p.setJointMotorControl2(
            bodyUniqueId=env.robot_id,
            jointIndex=left_back_wheel,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=left_side_speed,
            force=env.move_force,
            physicsClientId=env.client_id
        )

        # 打印调试信息
        if env.verbose >= 2:
            if env.current_step % 20 == 0:
                pos, _ = p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id)
                lin_vel, ang_vel = p.getBaseVelocity(env.robot_id, physicsClientId=env.client_id)
                print(f"动作: forward={forward:.2f}, turn={turn:.2f}")
                print(f"轮速: 右侧={right_side_speed:.2f}, 左侧={left_side_speed:.2f}")
                print(f"机器人位置: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
                print(f"线速度: {lin_vel}, 角速度: {ang_vel}")
