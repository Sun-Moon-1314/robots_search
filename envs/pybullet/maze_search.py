# -*- coding: utf-8 -*-
"""
@File    : maze_env.py
@Author  : zhangjian
@Desc    : 迷宫环境实现
"""

import gymnasium as gym
import time
import pybullet_data
import logging
import random

from envs.pybullet.maze_builder import MazeBuilder
from custom_design.maze_search_reward import compute_reward
from custom_design.maze_search_obs import *

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MazeEnv(gym.Env):
    """机器人在迷宫中寻找小球的强化学习环境"""

    def __init__(self, maze_size=(7, 7), render_mode=None, max_steps=1000, verbose=False, env_config=None,
                 is_eval=False):
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
        self.eval_mode = is_eval
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
        if verbose >= 2:
            logger.info(f"速度因子: {self.speed_factor}, 动力因子: {self.max_force}")

        # 添加对随机位置的支持
        self.use_random_positions = False
        self.random_positions = None
        self.env_config = env_config if env_config is not None else {}  # 确保env_config不为None

        # 如果环境配置中包含随机位置，则设置相关属性
        if "random_positions" in self.env_config:
            self.use_random_positions = True
            self.random_positions = self.env_config["random_positions"]
            if self.verbose:
                print(f"启用随机位置模式，可选位置: {self.random_positions}")
        else:
            self.use_random_positions = False
            self.random_positions = []
        # 设置起点和目标位置
        self.start_pos = (1, 1)  # 机器人的起始位置，以网格坐标表示
        # 设置球位置，如果配置中有ball_pos则使用，否则为None（后续会设置）

        # 设置球位置
        if self.env_config.get("ball_pos") is not None:
            self.ball_pos = self.env_config.get("ball_pos")
        else:
            self.ball_pos = (1, 1)

        # 简化处理：goal_pos 直接等于 ball_pos
        self.goal_pos = self.ball_pos
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
        # 朝向，速度等22维度
        self.observation_space = define_observation_space()

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
        print(f"生成迷宫中的球位置: {self.ball_pos}")
        print(f"随机球列表: {self.random_positions}")

        # 设置球位置 - 从random_positions中随机选择一个位置
        if (hasattr(self, 'use_random_positions')
            and self.use_random_positions
            and hasattr(self, 'random_positions') and self.random_positions):
            self.ball_pos = random.choice(self.random_positions)
        elif self.ball_pos is None:
            self.ball_pos = (1, 1)
        print(f"设置球位置后: {self.ball_pos}")
        # 确保球位置在迷宫范围内且是空地
        if (0 <= self.ball_pos[0] < self.maze_size[0] and
                0 <= self.ball_pos[1] < self.maze_size[1]):
            # 确保该位置是空地，如果不是则将其设为空地
            if self.maze_data[self.ball_pos[0]][self.ball_pos[1]] != 0:
                self.maze_data[self.ball_pos[0]][self.ball_pos[1]] = 0
        else:
            # 如果位置超出范围，则使用右下角附近的位置
            self.ball_pos = (self.maze_size[0] - 2, self.maze_size[1] - 2)
            # 确保该位置是空地
            self.maze_data[self.ball_pos[0]][self.ball_pos[1]] = 0

        # 简化处理：goal_pos 直接等于 ball_pos
        self.goal_pos = self.ball_pos

    def build_maze(self):
        """构建迷宫环境"""
        # 使用迷宫构建器构建环境
        self.maze_builder.build_ground()
        self.maze_builder.build_walls(self.maze_data)

        # 放置机器人
        self.robot_id = self.maze_builder.place_robot(self.start_pos)

        # 如果启用了随机位置且有可用的随机位置列表
        if (hasattr(self, 'use_random_positions') and self.use_random_positions
            and hasattr(self, 'random_positions') and self.random_positions) and self.ball_pos is None:
            print(f"构建迷宫时随机选择球位置：{self.ball_pos}")

        # 如果目标位置设置为跟随球位置或未设置
        self.goal_pos = self.ball_pos

        # 放置目标球体
        self.ball_id = self.maze_builder.place_goal(self.ball_pos)  # 注意这里使用ball_pos而不是goal_pos
        # print(f"球的ID:{self.ball_id}")
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

    def _get_obs(self):
        """获取当前观察状态，包含平衡状态信息"""
        return get_observation(self)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)  # 确保random模块也使用相同的种子

        # 重置环境状态
        self.current_step = 0
        self.done = False

        # 如果目标位置设置为跟随球位置
        self.goal_pos = self.ball_pos
        if self.verbose >= 2:
            print(f"-----{self.current_step}-----")
            print(f"球的位置:{self.ball_pos}")
            print(f"目标位置:{self.goal_pos}")

        # 如果是首次重置或者需要完全重建环境
        if not self.initialized:
            # 构建迷宫
            self.build_maze()
            self.initialized = True
        else:
            # 后续重置只需重置对象位置
            self.maze_builder.reset_robot_position(self.start_pos)

            # 重置目标位置
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
            if self.verbose >= 2:
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
        """获取当前环境状态的信息，仅包含不作为观察空间一部分的额外监控信息"""
        info = {
            # 任务相关指标
            'found_ball': False,  # 是否找到球
            'success': False,  # 是否成功完成任务
            'timeout': False,  # 是否超时
            'progress': 0.0,  # 任务进度 (0-1)

            # 碰撞信息
            'collision': False,  # 是否发生碰撞
            'ball_collision': False,  # 是否与球碰撞

            # 记录当前动作
            'actions': np.zeros(2)  # 当前执行的动作
        }

        # 添加当前动作到info
        if hasattr(self, 'current_action'):
            info['actions'] = self.current_action

        # 计算到球的距离和任务相关信息
        if self.robot_id is not None and self.ball_id is not None:
            robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
            ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client_id)

            # 计算到球的距离 (这个值在get_obs中已经计算过，但在info中也需要)
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

            # 使用碰撞检测判断是否找到球
            if ball_collision or bool(distance_to_ball <= 0.60):
                info['found_ball'] = True
                info['success'] = True

            # 检查一般碰撞
            info['collision'] = self._check_collision()

            # 计算任务进度
            progress = 1.0 - (distance_to_ball / self.initial_distance_to_ball)
            info['progress'] = max(0.0, min(1.0, progress))  # 限制在0到1之间

        # 检查是否超时
        if self.current_step >= self.max_steps:
            info['timeout'] = True

        return info

    def _check_done(self, observation, info):
        """检查是否终止"""
        # 1. 找到球
        if info['found_ball']:
            return True

        # 2. 超时
        if info['timeout']:
            return True

        # 3. 倾斜过大（失去平衡）
        MAX_TILT = 0.7  # 约40度
        roll = observation[5]
        pitch = observation[6]
        tilt_angle = math.sqrt(roll ** 2 + pitch ** 2)  # 计算总倾斜角度
        if tilt_angle > MAX_TILT:
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
        reward = compute_reward(self, observation, info)

        # 检查是否终止
        done = self._check_done(observation, info)

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
