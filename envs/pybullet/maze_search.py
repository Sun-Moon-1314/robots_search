# -*- coding: utf-8 -*-
"""
@File    : maze_env.py
@Author  : zhangjian
@Desc    : 迷宫环境实现
"""

import gymnasium as gym
import pybullet as p
import numpy as np
import math
import time
from gymnasium import spaces
import pybullet_data
import logging

from envs.pybullet.maze_builder import MazeBuilder
from reward_design.maze_search_reward import compute_reward

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MazeEnv(gym.Env):
    """机器人在迷宫中寻找小球的强化学习环境"""

    def __init__(self, maze_size=(7, 7), render_mode=None, max_steps=1000, verbose=False, **kwargs):
        """
        初始化迷宫环境

        Args:
            maze_size: 迷宫大小，(width, height)元组
            render_mode: 渲染模式，"human"表示可视化，None表示无可视化
            max_steps: 每个回合的最大步数
            verbose: 是否打印详细信息
        """
        super(MazeEnv, self).__init__()
        # 环境参数
        self.eval_mode = kwargs.get('is_eval', False)
        self.verbose = verbose
        self.maze_size = maze_size  # 迷宫的大小，表示为 N x N 的网格
        self.max_steps = max_steps  # 每个回合的最大步数，超过此步数环境会自动结束
        self.current_step = 0  # 当前步数计数器，用于追踪回合进度
        self.cell_size = 1.0  # 迷宫中每个单元格的大小，单位为米
        self.render_mode = render_mode  # 渲染模式，"human"表示可视化，None表示无可视化
        self.client_id = None  # PyBullet客户端ID，用于标识PyBullet实例
        self.initialized = False  # 环境初始化标志，表示环境是否已完成初始化
        self.curriculum_phase = 3  # 默认为最终阶段

        if self.curriculum_phase == 1:
            self.speed_factor = 1.0
            self.max_force = 1.0
        elif self.curriculum_phase == 2:
            self.speed_factor = 1.2
            self.max_force = 1.2
        elif self.curriculum_phase == 3:
            self.speed_factor = 1.5
            self.max_force = 1.5
        if verbose:
            logger.info(f"速度因子: {self.speed_factor}, 动力因子: {self.max_force}")

        # 设置起点和目标位置
        self.start_pos = (1, 1)  # 机器人的起始位置，以网格坐标表示
        self.goal_pos = None  # 目标位置，将在generate_maze方法中设置
        self.maze_data = None  # 迷宫数据，存储墙壁和通道的布局

        # 记录上一步的距离，用于计算奖励
        self.prev_distance_to_ball = None  # 上一步机器人到目标球的距离，用于计算进度奖励

        # 机器人和球的ID
        self.robot_id = None  # PyBullet中R2D2机器人模型的唯一标识符
        self.ball_id = None  # PyBullet中目标球体的唯一标识符
        self.maze_builder = None  # 迷宫构建器

        # PyBullet 初始化
        if render_mode == "human":
            self.client_id = p.connect(p.GUI)  # 连接到PyBullet的GUI模式，可视化物理模拟
        else:
            self.client_id = p.connect(p.DIRECT)  # 连接到PyBullet的DIRECT模式，无可视化，更快

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 设置额外的搜索路径，用于加载模型

        # 动作空间：前进速度和转向角度
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),  # 动作的最小值：[前进速度最小值, 转向角度最小值]
            high=np.array([1.0, 1.0]),  # 动作的最大值：[前进速度最大值, 转向角度最大值]
            dtype=np.float32  # 数据类型为32位浮点数
        )

        # 观察空间：机器人位置(x,y)、机器人朝向(yaw)、球位置(x,y)、
        # 以及激光扫描数据(8个方向的距离)和平衡状态(roll, pitch, roll_rate, pitch_rate)
        self.observation_space = spaces.Box(
            low=-np.inf,  # 观察值的最小值，设为负无穷
            high=np.inf,  # 观察值的最大值，设为正无穷
            shape=(17,),  # 观察空间维度：[robot_x, robot_y, robot_yaw, ball_x, ball_y,
            # roll, pitch, roll_rate, pitch_rate, 8个激光数据]
            dtype=np.float32  # 数据类型为32位浮点数
        )

        # 设置物理引擎参数
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / 240.0,  # 物理模拟的时间步长，设置为1/240秒以提高稳定性
            numSolverIterations=50,  # 物理求解器的迭代次数，增加以提高精度
            numSubSteps=4,  # 每个模拟步骤的子步骤数，增加以提高稳定性
            physicsClientId=self.client_id  # 指定PyBullet客户端ID
        )
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)  # 设置标准重力加速度

        # 初始化迷宫构建器
        self.maze_builder = MazeBuilder(self.client_id, maze_size, self.cell_size)

        # 生成迷宫
        self.generate_maze()  # 调用方法生成迷宫结构和目标位置

    def generate_maze(self):
        """生成固定迷宫并设置固定的球位置"""
        # 使用迷宫构建器生成迷宫数据
        self.maze_data = self.maze_builder.generate_maze_data()

        # 设置固定的球位置（指定具体坐标）
        fixed_ball_pos = (1, 5)

        # 确保指定的位置在迷宫范围内
        if (0 <= fixed_ball_pos[0] < self.maze_size[0] and
                0 <= fixed_ball_pos[1] < self.maze_size[1]):
            # 确保该位置是空地，如果不是则将其设为空地
            if self.maze_data[fixed_ball_pos[0]][fixed_ball_pos[1]] != 0:
                self.maze_data[fixed_ball_pos[0]][fixed_ball_pos[1]] = 0

            self.ball_pos = fixed_ball_pos
        else:
            # 如果指定位置超出范围，则使用右下角附近的位置
            self.ball_pos = (self.maze_size[0] - 2, self.maze_size[1] - 2)
            # 确保该位置是空地
            self.maze_data[self.ball_pos[0]][self.ball_pos[1]] = 0

        self.goal_pos = self.ball_pos  # 设置goal_pos与ball_pos一致

    def build_maze(self):
        """构建迷宫环境"""
        # 使用迷宫构建器构建环境
        self.maze_builder.build_ground()
        self.maze_builder.build_walls(self.maze_data)
        self.robot_id = self.maze_builder.place_robot(self.start_pos)
        self.ball_id = self.maze_builder.place_goal(self.goal_pos)

        # 可以调整机器人的物理属性
        if self.robot_id is not None:
            # 增加机器人底部的摩擦系数，提高稳定性
            p.changeDynamics(
                self.robot_id,
                -1,  # 基础链接
                lateralFriction=0.8,  # 增加摩擦
                spinningFriction=0.1,  # 增加旋转摩擦
                rollingFriction=0.1,  # 增加滚动摩擦
                restitution=0.1,  # 降低弹性
                linearDamping=0.1,  # 增加线性阻尼
                angularDamping=0.1,  # 增加角度阻尼
                physicsClientId=self.client_id
            )

    def _get_laser_scan(self):
        """获取8个方向的激光扫描数据"""
        # 默认返回8个方向的最大距离
        default_scan = [5.0] * 8

        if self.robot_id is None:
            return default_scan

        try:
            robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
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
                results = p.rayTest(ray_start, ray_end, physicsClientId=self.client_id)

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

    def _get_obs(self):
        """获取当前观察状态，包含平衡状态信息"""
        # 初始化默认观察值（17个值）
        default_obs = np.zeros(17, dtype=np.float32)

        if self.robot_id is None:
            return default_obs

        try:
            # 获取机器人位置和朝向
            robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
            robot_x, robot_y, _ = robot_pos
            roll, pitch, yaw = p.getEulerFromQuaternion(robot_orn)

            # 获取线性速度和角速度（用于平衡状态分析）
            linear_vel, angular_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.client_id)
            roll_rate, pitch_rate = angular_vel[0], angular_vel[1]

            # 获取球的位置
            if self.ball_id is not None:
                ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client_id)
                ball_x, ball_y, _ = ball_pos
            else:
                ball_x, ball_y = 0, 0

            # 获取激光扫描数据
            laser_data = self._get_laser_scan()

            # 组合所有观察数据，包括平衡状态
            obs = np.array([
                robot_x, robot_y, yaw, ball_x, ball_y,  # 5个基本状态
                roll, pitch, roll_rate, pitch_rate,  # 4个平衡状态相关的值
                *laser_data  # 8个激光扫描数据
            ], dtype=np.float32)

            # 确保维度正确
            if obs.shape != (17,):
                print(f"警告: 观察值维度不正确: {obs.shape}，使用默认值")
                return default_obs

            if self.verbose:
                print(f"机器人位置：({obs[0]}, {obs[1]})")
                print(f"机器人水平面内朝向角：{obs[2]}")
                print(f"小球位置：({obs[3]}, {obs[4]})")
                print(f"机器人左右倾斜角：{obs[5]}")
                print(f"机器人左右倾斜角速度：{obs[7]}")
                print(f"机器人前后倾斜角：{obs[6]}")
                print(f"机器人前后倾斜角速度：{obs[8]}")

            return obs

        except Exception as e:
            print(f"获取观察值时出错: {e}")
            return default_obs

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # 重置环境状态
        self.current_step = 0
        self.done = False

        # 如果是首次重置或者需要完全重建环境
        if not self.initialized:
            # 构建迷宫
            self.build_maze()
            self.initialized = True
        else:
            # 后续重置只需重置对象位置
            self.maze_builder.reset_robot_position(self.start_pos)
            self.maze_builder.reset_goal_position(self.goal_pos)

        # 重置上一步距离
        self.prev_distance_to_ball = None

        # 计算初始距离
        if self.robot_id is not None and self.ball_id is not None:
            robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
            self.prev_distance_to_ball = math.sqrt(
                (robot_pos[0] - ball_pos[0]) ** 2 +
                (robot_pos[1] - ball_pos[1]) ** 2
            )

        # 确保小车初始状态是平衡的
        if self.robot_id is not None:
            # 获取当前位置和朝向
            current_pos, current_orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)

            # 重置为平衡状态（保持位置，但重置朝向为平衡状态）
            _, _, yaw = p.getEulerFromQuaternion(current_orn)
            balanced_orn = p.getQuaternionFromEuler([0, 0, yaw])

            # 重置机器人位置和朝向
            p.resetBasePositionAndOrientation(
                self.robot_id,
                current_pos,
                balanced_orn,
                physicsClientId=self.client_id
            )

            # 重置速度为零
            p.resetBaseVelocity(
                self.robot_id,
                linearVelocity=[0, 0, 0],
                angularVelocity=[0, 0, 0],
                physicsClientId=self.client_id
            )

        # 设置可鼠标控制的视角（GUI模式下）
        if self.render_mode == "human":
            self.maze_builder.setup_camera()

        # 获取初始观察
        observation = self._get_obs()

        # 返回观察和空信息字典
        return observation, {}

    def _check_collision(self):
        """检查是否发生碰撞"""
        if self.robot_id is not None:
            # 获取与机器人接触的物体
            contact_points = p.getContactPoints(bodyA=self.robot_id, physicsClientId=self.client_id)

            # 过滤掉与地面的接触
            for contact in contact_points:
                # 如果接触的物体不是地面且不是球，则认为是碰撞
                if contact[2] != self.maze_builder.plane_id and contact[2] != self.ball_id:
                    return True

        return False

    def _apply_action(self, action):
        """应用动作到机器人"""
        # 应用动作
        if self.robot_id is not None:
            # 对动作进行平滑处理，避免突变
            if not hasattr(self, 'prev_action'):
                self.prev_action = np.zeros_like(action)

            # 平滑系数 - 值越小平滑效果越强
            smoothing = 0.3

            # 平滑动作变化
            smoothed_action = smoothing * np.array(action) + (1 - smoothing) * self.prev_action
            self.prev_action = smoothed_action

            # 解析动作（前进速度和转向）
            forward = float(smoothed_action[0]) * 0.4  # 降低最大速度
            turn = float(smoothed_action[1]) * 0.25  # 降低最大转向

            # 保存当前动作到info中供奖励函数使用
            self.current_action = np.array([forward, turn])

            # 根据打印的关节信息，R2D2有4个轮子
            right_front_wheel = 2  # right_front_wheel_joint
            right_back_wheel = 3  # right_back_wheel_joint
            left_front_wheel = 6  # left_front_wheel_joint
            left_back_wheel = 7  # left_back_wheel_joint

            # 计算轮子速度
            speed_factor = 20.0 * self.speed_factor  # 降低速度因子
            right_side_speed = speed_factor * (forward - turn)
            left_side_speed = speed_factor * (forward + turn)

            # 设置最大力以确保足够的动力
            max_force = 60.0 * self.max_force  # 降低最大力矩

            # 控制所有四个轮子
            # 右侧两个轮子
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=right_front_wheel,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=right_side_speed,
                force=max_force,
                physicsClientId=self.client_id
            )

            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=right_back_wheel,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=right_side_speed,
                force=max_force,
                physicsClientId=self.client_id
            )

            # 左侧两个轮子
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=left_front_wheel,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=left_side_speed,
                force=max_force,
                physicsClientId=self.client_id
            )

            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=left_back_wheel,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=left_side_speed,
                force=max_force,
                physicsClientId=self.client_id
            )

            # 打印调试信息
            if self.verbose:
                if self.current_step % 20 == 0:
                    pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
                    lin_vel, ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.client_id)
                    print(f"动作: forward={forward:.2f}, turn={turn:.2f}")
                    print(f"轮速: 右侧={right_side_speed:.2f}, 左侧={left_side_speed:.2f}")
                    print(f"机器人位置: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
                    print(f"线速度: {lin_vel}, 角速度: {ang_vel}")

    def _check_ball_collision(self):
        """检查机器人是否与球碰撞"""
        if self.robot_id is None or self.ball_id is None:
            return False

        # 获取机器人和球之间的接触点
        contact_points = p.getContactPoints(
            bodyA=self.robot_id,
            bodyB=self.ball_id,
            physicsClientId=self.client_id
        )

        # 如果有接触点，则表示发生了碰撞
        return len(contact_points) > 0

    def _get_info(self):
        """获取当前环境状态的信息，增加侧向速度和移动方向一致性计算，并使用碰撞检测判断是否找到球"""
        info = {
            'distance_to_ball': float('inf'),  # 默认值
            'found_ball': False,  # 默认值
            'collision': False,  # 默认值
            'ball_collision': False,  # 与球的碰撞状态（新增）
            'timeout': False,  # 是否超时
            'success': False,  # 是否成功
            'balance_angle': 0.0,  # 平衡角度
            'actions': np.zeros(2),  # 默认动作值
            'forward_velocity': 0.0,  # 前进速度
            'lateral_velocity': 0.0,  # 侧向速度
            'movement_orientation_alignment': 0.0,  # 移动方向与朝向一致性
            'robot_orientation': 0.0,  # 机器人朝向
            'target_direction': 0.0,  # 目标方向
            'orientation_alignment': 0.0  # 朝向与目标方向的一致性
        }

        # 添加当前动作到info
        if hasattr(self, 'current_action'):
            info['actions'] = self.current_action

        # 计算到球的距离和方向信息
        if self.robot_id is not None and self.ball_id is not None:
            robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
            ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client_id)

            # 计算到球的距离
            distance_to_ball = math.sqrt(
                (robot_pos[0] - ball_pos[0]) ** 2 +
                (robot_pos[1] - ball_pos[1]) ** 2
            )
            info['distance_to_ball'] = distance_to_ball

            # 记录初始距离（如果尚未记录）
            if not hasattr(self, 'initial_distance_to_ball'):
                self.initial_distance_to_ball = distance_to_ball
            info['initial_distance_to_ball'] = self.initial_distance_to_ball

            # 检查是否与球碰撞
            ball_collision = self._check_ball_collision()
            info['ball_collision'] = ball_collision

            # 使用碰撞检测判断是否找到球，而不是固定距离阈值
            if ball_collision or bool(distance_to_ball <= 0.60):
                info['found_ball'] = True
                info['success'] = True

            # 计算平衡角度
            roll, pitch, yaw = p.getEulerFromQuaternion(robot_orn)
            tilt_angle = math.sqrt(roll ** 2 + pitch ** 2)
            info['balance_angle'] = tilt_angle

            # 保存机器人朝向（以yaw角表示）
            info['robot_orientation'] = yaw

            # 计算目标方向（从机器人到球的方向）
            target_dx = ball_pos[0] - robot_pos[0]
            target_dy = ball_pos[1] - robot_pos[1]
            target_direction = math.atan2(target_dy, target_dx)
            info['target_direction'] = target_direction

            # 计算朝向与目标方向的一致性（-1到1，1表示完全一致）
            orientation_diff = (target_direction - yaw + math.pi) % (2 * math.pi) - math.pi
            orientation_alignment = math.cos(orientation_diff)
            info['orientation_alignment'] = orientation_alignment

            # 获取线性和角速度
            linear_vel, angular_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.client_id)

            # 计算前向向量（机器人朝向的单位向量）
            # 从四元数获取旋转矩阵的前两列（对应x和y方向）
            rot_matrix = p.getMatrixFromQuaternion(robot_orn)
            forward_x, forward_y = rot_matrix[0], rot_matrix[3]  # 矩阵的第一列

            # 归一化前向向量
            forward_norm = math.sqrt(forward_x ** 2 + forward_y ** 2)
            if forward_norm > 0:
                forward_x /= forward_norm
                forward_y /= forward_norm

            # 计算速度向量在水平面的投影
            vx, vy = linear_vel[0], linear_vel[1]
            velocity_magnitude = math.sqrt(vx ** 2 + vy ** 2)

            # 计算前进速度（沿着朝向方向的速度分量）- 点积
            forward_velocity = vx * forward_x + vy * forward_y
            info['forward_velocity'] = forward_velocity

            # 计算侧向速度（垂直于朝向方向的速度分量）
            # 使用叉积的模长来计算
            lateral_velocity = abs(vx * forward_y - vy * forward_x)
            info['lateral_velocity'] = lateral_velocity

            # 计算移动方向与朝向的一致性
            if velocity_magnitude > 0.1:  # 只在有明显移动时计算
                movement_dir_x = vx / velocity_magnitude
                movement_dir_y = vy / velocity_magnitude
                # 点积计算一致性（-1到1，1表示完全一致）
                movement_alignment = forward_x * movement_dir_x + forward_y * movement_dir_y
                info['movement_orientation_alignment'] = movement_alignment
            else:
                info['movement_orientation_alignment'] = 0.0  # 静止时默认为0

            # 保存线性和角速度信息
            info['linear_velocity'] = linear_vel
            info['angular_velocity'] = angular_vel

            # 检查碰撞
            info['collision'] = self._check_collision()

            # 计算任务进度
            progress = 1.0 - (distance_to_ball / self.initial_distance_to_ball)
            info['progress'] = max(0.0, min(1.0, progress))  # 限制在0到1之间

        # 检查是否超时
        if self.current_step >= self.max_steps:
            info['timeout'] = True

        return info

    def _check_done(self, info):
        """检查是否终止"""
        # 1. 找到球
        if info['found_ball']:
            return True

        # 2. 超时
        if info['timeout']:
            return True

        # 3. 倾斜过大（失去平衡）
        MAX_TILT = 0.7  # 约40度
        if info['balance_angle'] > MAX_TILT:
            return True

        # 4. 其他终止条件（如掉出地图）
        if self.robot_id is not None:
            robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
            # 检查是否掉出地图
            if robot_pos[2] < -1.0:  # z坐标小于-1表示掉出地图
                return True

        return False

    def step(self, action):
        self.current_step += 1

        # 应用动作
        self._apply_action(action)

        # 模拟多步物理以确保动作生效
        for _ in range(10):  # 模拟多步，但不要太多以免影响性能
            p.stepSimulation(physicsClientId=self.client_id)
            if self.render_mode == "human":
                time.sleep(0.001)

        # 获取新的观察
        observation = self._get_obs()

        # 获取信息字典
        info = self._get_info()

        # 计算奖励
        reward = compute_reward(self, info)

        # 检查是否终止
        done = self._check_done(info)

        # 返回observation, reward, done, truncated, info
        return observation, reward, done, False, info

    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            # 调整相机视角
            if self.robot_id is not None:
                robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
                target_pos = [robot_pos[0], robot_pos[1], 0]

                view_matrix = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=target_pos,
                    distance=10.0,
                    yaw=45,
                    pitch=-30,
                    roll=0,
                    upAxisIndex=2
                )

                proj_matrix = p.computeProjectionMatrixFOV(
                    fov=60,
                    aspect=1.0,
                    nearVal=0.1,
                    farVal=100.0
                )

                (_, _, px, _, _) = p.getCameraImage(
                    width=640,
                    height=480,
                    viewMatrix=view_matrix,
                    projectionMatrix=proj_matrix
                )

                return px

        return None

    def close(self):
        if self.client_id is not None:
            try:
                p.disconnect(self.client_id)
            except Exception as e:
                print(f"Error during disconnect: {e}")
            self.client_id = None

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            print(f"Exception ignored in __del__: {e}")
