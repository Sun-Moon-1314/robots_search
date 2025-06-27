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

from custom_design.maze_search_action import apply_action
from custom_design.maze_search_done import check_done
from custom_design.maze_search_info import get_info
from envs.pybullet.maze_builder import MazeBuilder
from custom_design.maze_search_reward import compute_reward
from custom_design.maze_search_obs import *

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MazeEnv(gym.Env):
    """机器人在迷宫中寻找小球的强化学习环境"""

    def __init__(self, maze_size=(7, 7),
                 render_mode=None,
                 curriculum_phase=None,
                 max_steps=1000,
                 default_max_steps=1000,
                 steps_per_grid=100,
                 min_steps=200,
                 max_allowed_steps=2000,
                 verbose=False,
                 env_config=None,
                 is_eval=False,
                 seed=42):
        """
        初始化迷宫环境

        Args:
        maze_size: 迷宫大小，(width, height)元组
        render_mode: 渲染模式，"human"表示可视化，None表示无可视化
        max_steps: 每个回合的最大步数（初始值，后续可能动态调整）
        default_max_steps: 默认最大步数，当无法计算距离时使用
        steps_per_grid: 每个网格单位距离对应的步数，用于动态计算 max_steps
        min_steps: 动态计算 max_steps 时的最小步数限制
        max_allowed_steps: 动态计算 max_steps 时的最大步数限制，基于 (7,7) 转一圈的步数
        verbose: 是否打印详细信息
        env_config: 环境配置字典
        is_eval: 是否为评估模式
        seed: 随机种子
        """
        super(MazeEnv, self).__init__()
        # 随机种子初始化
        self.seed_value = seed  # 存储种子值
        random.seed(seed)
        np.random.seed(seed)
        # 环境参数
        self.initial_distance_to_ball = None
        self.eval_mode = is_eval
        self.verbose = verbose
        self.maze_size = maze_size  # 迷宫的大小，表示为 N x N 的网格
        self.max_steps = max_steps  # 每个回合的最大步数，超过此步数环境会自动结束
        self.current_step = 0  # 当前步数计数器，用于追踪回合进度
        self.cell_size = 1.0  # 迷宫中每个单元格的大小，单位为米
        self.render_mode = render_mode  # 渲染模式，"human"表示可视化，None表示无可视化
        self.client_id = None  # PyBullet客户端ID，用于标识PyBullet实例
        self.initialized = False  # 环境初始化标志，表示环境是否已完成初始化
        self.curriculum_phase = curriculum_phase  # 默认为最终阶段
        self.current_done = False
        # 动态步数相关参数
        self.default_max_steps = default_max_steps  # 默认最大步数
        self.steps_per_grid = steps_per_grid  # 每个网格单位允许的步数
        self.min_steps = min_steps  # 最小步数限制
        self.max_allowed_steps = max_steps if max_steps > max_allowed_steps else max_allowed_steps  # 最大步数限制
        self.max_steps = default_max_steps  # 初始化为默认值

        self.move_speed = 30.0
        self.move_force = 80.0
        self.prev_robot_pos = None
        self.avg_speed = 0.0
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

        # 添加墙体位置列表属性
        self.wall_positions = []
        # 记录上一步的角度差，用于方向探索奖励计算
        self.prev_angle_diff = None
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
        # 在环境类或评估代码中添加朝向累积
        if not hasattr(self, 'manual_yaw'):
            self.manual_yaw = 0.0  # 初始化朝向为0
        # 观察空间
        self.observation_space = define_observation_space()
        self.time_step = 1.0 / 240.0
        # 设置物理引擎参数
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.time_step,  # 物理模拟的时间步长，设置为1/240秒以提高稳定性
            numSolverIterations=50,  # 物理求解器的迭代次数，增加以提高精度
            numSubSteps=4,  # 每个模拟步骤的子步骤数，增加以提高稳定性
            physicsClientId=self.client_id  # 指定PyBullet客户端ID
        )
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)  # 设置标准重力加速度

        self.max_tilt_angle = self.env_config.get("max_tilt_angle", 0.7)
        self.tilt_angle = 0.0
        self.distance_to_target = 0.0
        self.movement_alignment = 0.0
        # 在环境类中添加相机状态
        self.camera_current_pos = [0, 0, 0]
        self.camera_current_yaw = 0
        self.camera_current_pitch = -30
        self.camera_current_distance = None
        # 初始化迷宫构建器
        self.maze_builder = MazeBuilder(self.client_id, maze_size, self.cell_size)

        # 生成迷宫
        self.generate_maze()  # 调用方法生成迷宫结构和目标位置

        # 新增：保存最近的原始观察值
        self.last_raw_obs = None

    def generate_maze(self):
        """生成固定迷宫并设置固定的球位置"""
        # 使用迷宫构建器生成迷宫数据
        self.maze_data = self.maze_builder.generate_maze_data()

        # 提取并存储墙体位置
        self.wall_positions = []
        for i in range(self.maze_size[0]):
            for j in range(self.maze_size[1]):
                if self.maze_data[i][j] == 1:  # 假设1表示墙体
                    self.wall_positions.append((i, j))

        # 设置球位置 - 从random_positions中随机选择一个位置
        if (hasattr(self, 'use_random_positions')
                and self.use_random_positions
                and hasattr(self, 'random_positions') and self.random_positions):
            # 过滤掉墙体位置
            valid_positions = [pos for pos in self.random_positions if pos not in self.wall_positions]

            if valid_positions:
                self.ball_pos = random.choice(valid_positions)
            else:
                print("警告：没有有效的球位置（所有位置都是墙体）")
                self.ball_pos = (1, 1)  # 默认位置
        elif self.ball_pos is None:
            self.ball_pos = (1, 1)

        # 确保球位置在迷宫范围内且是空地
        if (0 <= self.ball_pos[0] < self.maze_size[0] and
                0 <= self.ball_pos[1] < self.maze_size[1]):
            # 确保该位置是空地，如果不是则将其设为空地
            if self.maze_data[self.ball_pos[0]][self.ball_pos[1]] != 0:
                self.maze_data[self.ball_pos[0]][self.ball_pos[1]] = 0
                # 更新墙体位置列表
                if self.ball_pos in self.wall_positions:
                    self.wall_positions.remove(self.ball_pos)
        else:
            # 如果位置超出范围，则使用右下角附近的位置
            self.ball_pos = (self.maze_size[0] - 2, self.maze_size[1] - 2)
            # 确保该位置是空地
            self.maze_data[self.ball_pos[0]][self.ball_pos[1]] = 0
            # 更新墙体位置列表
            if self.ball_pos in self.wall_positions:
                self.wall_positions.remove(self.ball_pos)

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

    def _update_max_steps(self):
        """
        根据起点到目标点的距离动态设置最大步数
        """
        # 获取起点和目标点的位置
        start_pos = None
        goal_pos = None

        if hasattr(self, 'robot_id') and self.robot_id is not None:
            start_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        else:
            start_pos = self.start_pos  # 使用预设的起点

        if hasattr(self, 'ball_id') and self.ball_id is not None:
            goal_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        else:
            goal_pos = self.goal_pos  # 使用预设的目标点

        # 确保两个位置都有效
        if start_pos is not None and goal_pos is not None:
            # 计算欧几里得距离
            grid_distance = math.sqrt(
                (start_pos[0] - goal_pos[0]) ** 2 +
                (start_pos[1] - goal_pos[1]) ** 2
            )

            # 计算曼哈顿距离（适用于网格环境）
            # grid_distance = np.linalg.norm(np.array(goal_pos) - np.array(start_pos))

            # 每个格子的步数（简化计算，固定值）
            # steps_per_grid = self.steps_per_grid if hasattr(self, 'steps_per_grid') else 50

            # 使用动态调整的 steps_per_grid，并考虑绕路增加 50% 步数
            calculated_steps = int(grid_distance * self.steps_per_grid * 1.5)
            self.max_steps = max(self.min_steps, min(calculated_steps, self.max_allowed_steps))

            if grid_distance < 1.0:  # 接近目标额外增加步数
                self.max_steps += 100
            # if self.verbose:
            #     print(f"动态调整后最大步数: {self.max_steps}, 每网格步数: {self.steps_per_grid}")

        else:
            # 如果无法获取位置，使用默认步数
            self.max_steps = self.default_max_steps if hasattr(self, 'default_max_steps') else 500
            if self.verbose:
                print(f"无法获取起点或目标点位置，使用默认最大步数: {self.max_steps}")

    def reset(self, seed=None, options=None):
        """
        重置环境
        :param seed:
        :param options:
        :return:
        """
        # 如果提供了新的种子，则使用它，否则使用初始化时的种子
        seed_to_use = seed if seed is not None else self.seed_value
        # 重新设置所有随机数生成器
        random.seed(seed_to_use)
        np.random.seed(seed_to_use)
        self.manual_yaw = 0
        # 重置环境状态
        self.current_step = 0
        self.done = False
        self.prev_observation = None
        # 重置角度差追踪变量
        self.prev_angle_diff = None
        # 如果启用了随机位置，在每次重置时重新随机选择球的位置
        if self.use_random_positions and self.random_positions:
            # 获取机器人当前位置（如果已初始化）
            robot_pos = None
            if hasattr(self, 'robot_id') and self.robot_id is not None:
                robot_pos_3d, _ = p.getBasePositionAndOrientation(self.robot_id)
                # 转换为2D网格坐标
                robot_pos = (int(robot_pos_3d[0]), int(robot_pos_3d[1]))

            # 过滤掉墙体位置和机器人位置
            valid_positions = []
            for pos in self.random_positions:
                if pos not in self.wall_positions and pos != robot_pos:
                    valid_positions.append(pos)
                    # print(f"墙体位置:{valid_positions}")

            if valid_positions:
                self.ball_pos = random.choice(valid_positions)
                # 更新目标位置以匹配球位置
                self.goal_pos = self.ball_pos
                if self.verbose >= 2:
                    print(
                        f"重置环境：随机选择球位置为 {self.ball_pos}，有效位置数量: {len(valid_positions)}/{len(self.random_positions)}")
            else:
                print("警告：没有有效的球位置（所有位置都是墙体或机器人位置）")
                # 保持原位置不变
                if self.verbose:
                    print(f"保持原球位置: {self.ball_pos}")

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

        # 计算初始距离
        if self.robot_id is not None and self.ball_id is not None:
            robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
            # print(f"*****球真正的初始位置:{ball_pos}*****")
            self.prev_distance_to_ball = math.sqrt(
                (robot_pos[0] - ball_pos[0]) ** 2 +
                (robot_pos[1] - ball_pos[1]) ** 2
            )

        # 确保初始距离记录
        self.initial_distance_to_ball = self.prev_distance_to_ball

        # 动态设置最大步数 - 根据起点到目标点的距离
        self._update_max_steps()

        # 设置可鼠标控制的视角（GUI模式下）
        if self.render_mode == "human":
            self.maze_builder.setup_camera()

        # 获取初始观察
        observation = self._get_obs()

        # 新增：保存原始观察值
        self.last_raw_obs = observation
        # 更新性能指标值 - 添加这几行
        self.tilt_angle = math.sqrt(observation[5] ** 2 + observation[6] ** 2)
        self.distance_to_target = math.sqrt(observation[3] ** 2 + observation[4] ** 2)
        self.movement_alignment = observation[27]

        # 返回观察和空信息字典
        return observation, {}

    def _update_orientation(self, turn):
        """更新机器人朝向基于转向动作"""
        yaw_change = turn * self.time_step * 10.0  # 旋转速度系数
        self.manual_yaw += yaw_change
        self.manual_yaw = np.arctan2(np.sin(self.manual_yaw), np.cos(self.manual_yaw))

        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        new_orientation = p.getQuaternionFromEuler([0, 0, self.manual_yaw])

        p.resetBasePositionAndOrientation(
            self.robot_id,
            posObj=robot_pos,
            ornObj=new_orientation,
            physicsClientId=self.client_id
        )

    def step(self, action):
        """
        环境步进
        :param action: 动作
        :return: observation, reward, done, False, info
        """
        self.current_step += 1
        # 应用动作
        apply_action(self, action)
        # 更新朝向（从动作中获取 turn 值）
        turn = self.current_action[1] if hasattr(self, 'current_action') else action[1] * 0.25
        self._update_orientation(turn)
        # 模拟多步物理以确保动作生效
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.client_id)
            if self.render_mode == "human":
                time.sleep(1 / 240)

        # 更新速度（每步或每隔几步更新一次）
        if self.current_step % 5 == 0:  # 每 5 步更新一次速度
            self.update_robot_speed()
            self.adjust_steps_per_grid()
            self._update_max_steps()  # 动态更新 max_steps

        # 获取新的观察
        observation = self._get_obs()
        self.last_raw_obs = observation
        # 更新性能指标值
        self.tilt_angle = math.sqrt(observation[5] ** 2 + observation[6] ** 2)
        self.distance_to_target = math.sqrt(observation[3] ** 2 + observation[4] ** 2)
        self.movement_alignment = observation[27]

        # 获取信息字典
        info = get_info(self)

        # 检查是否终止
        done = check_done(self, observation, info)
        self.current_done = done
        # 计算奖励
        reward = compute_reward(self, observation, info)

        return observation, reward, done, False, info

    def get_raw_obs(self):
        """
        获取最近的原始观察值（未归一化）
        Returns:
            np.ndarray: 最近的原始观察值
        """
        return self.last_raw_obs

    def update_camera_smooth(self, target_pos, target_yaw, target_pitch, smoothing=0.1):
        """平滑更新相机位置和角度"""
        # 位置平滑过渡
        self.camera_current_pos[0] += smoothing * (target_pos[0] - self.camera_current_pos[0])
        self.camera_current_pos[1] += smoothing * (target_pos[1] - self.camera_current_pos[1])
        self.camera_current_pos[2] += smoothing * (target_pos[2] - self.camera_current_pos[2])

        # 角度平滑过渡
        self.camera_current_yaw += smoothing * (target_yaw - self.camera_current_yaw)
        self.camera_current_pitch += smoothing * (target_pitch - self.camera_current_pitch)

        return self.camera_current_pos, self.camera_current_yaw, self.camera_current_pitch

    def update_robot_speed(self):
        """
        计算机器人当前实际移动速度（米/秒）
        """
        if not hasattr(self, 'prev_robot_pos'):
            self.prev_robot_pos = None
            self.avg_speed = 0.0
            return 0.0

        current_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        if self.prev_robot_pos is not None:
            # 计算两步之间的距离
            distance = math.sqrt(
                (current_pos[0] - self.prev_robot_pos[0]) ** 2 +
                (current_pos[1] - self.prev_robot_pos[1]) ** 2
            )
            # 时间间隔：step 函数中执行 10 次物理模拟，每次 1/240 秒
            time_interval = 10 * (1.0 / 240.0)  # 约 0.0417 秒
            # 计算速度（米/秒）
            speed = distance / time_interval if time_interval > 0 else 0.0
            # 平滑速度：使用移动平均，避免突变
            smoothing = 0.2
            self.avg_speed = smoothing * speed + (1 - smoothing) * (
                self.avg_speed if hasattr(self, 'avg_speed') else 0.0)
        else:
            self.avg_speed = 0.0

        self.prev_robot_pos = current_pos
        return self.avg_speed

    def adjust_steps_per_grid(self):
        """
        根据机器人移动速度动态调整每个格子的步数限制
        """
        # 获取当前平均速度
        current_speed = self.avg_speed if hasattr(self, 'avg_speed') else 0.0
        # 基准速度：假设 1.0 米/秒 为正常速度
        base_speed = 1.0
        # 步数范围
        min_steps_per_grid = 50  # 速度快时最小步数
        max_steps_per_grid = 200  # 速度慢时最大步数
        base_steps_per_grid = 100  # 基准步数

        if current_speed <= 0.01:  # 几乎不动
            self.steps_per_grid = max_steps_per_grid
        else:
            # 速度与基准速度的比值，决定步数调整
            speed_ratio = current_speed / base_speed
            # 速度快时减少步数，速度慢时增加步数
            self.steps_per_grid = base_steps_per_grid * (1.0 / max(0.2, speed_ratio))
            # 限制在范围内
            self.steps_per_grid = max(min_steps_per_grid, min(max_steps_per_grid, int(self.steps_per_grid)))

        # if self.verbose and self.current_step % 20 == 0:
        #     print(f"当前速度: {current_speed:.2f} 米/秒, 调整后每网格步数: {self.steps_per_grid}")

        return self.steps_per_grid

    def get_performance_metrics(self):
        """
        获取当前环境状态下的性能指标

        Returns:
            dict: 包含各项性能指标的字典
        """
        # 获取info以检查是否成功
        return {
            "max_steps": self.current_step,  # 当前步数
            "tilt_angle": self.tilt_angle,  # 当前倾斜角度
            "distance_to_target": self.distance_to_target,  # 当前到目标的距离
            "movement_alignment": self.movement_alignment,  # 当前移动方向与朝向的一致性
        }

    def get_max_expected_change(self):
        """
        计算或获取每步的最大预期距离变化。
        如果 env_config 中指定了值，则使用该值；否则基于环境参数计算。
        """
        if 'max_expected_change' in self.env_config:
            return self.env_config['max_expected_change']
        else:
            # 基于每网格步数计算，平均移动距离的 2 倍
            avg_step_distance = self.cell_size / self.steps_per_grid
            return avg_step_distance * 3  # 例如 0.04 米（4 厘米）

    def get_done(self):
        """
        获取current_done，防止与底层环境冲突
        :return: Bool
        """
        return self.current_done

    def render(self, mode="human", camera_type="follow", smooth=True):
        """渲染环境的高级版本"""
        if self.robot_id is None:
            return None
        current_distance = 0.0
        width, height = 640, 480
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)

        # 获取机器人朝向
        robot_orientation = p.getEulerFromQuaternion(robot_orn)
        robot_yaw = math.degrees(robot_orientation[2])

        # 根据相机类型设置目标参数
        if camera_type == "follow":
            target_pos = [robot_pos[0], robot_pos[1], 0]
            target_yaw = robot_yaw + 180  # 在机器人后方
            target_pitch = -30
            target_distance = 5.0
        elif camera_type == "top_down":
            target_pos = [robot_pos[0], robot_pos[1], 0]
            target_yaw = 0
            target_pitch = -89
            target_distance = self.maze_size * 1.5 if hasattr(self, 'maze_size') else 15.0
        elif camera_type == "isometric":
            target_pos = [robot_pos[0], robot_pos[1], 0]
            target_yaw = 45
            target_pitch = -45
            target_distance = self.maze_size * 1.2 if hasattr(self, 'maze_size') else 12.0
        else:  # 默认
            target_pos = [robot_pos[0], robot_pos[1], 0]
            target_yaw = 45
            target_pitch = -30
            target_distance = 5.0

        # 平滑相机过渡
        if smooth and hasattr(self, 'camera_current_pos'):
            current_pos, current_yaw, current_pitch = self.update_camera_smooth(
                target_pos, target_yaw, target_pitch, smoothing=0.1
            )
        else:
            # 初始化相机状态或不使用平滑
            if not hasattr(self, 'camera_current_pos'):
                self.camera_current_pos = list(target_pos)
                self.camera_current_yaw = target_yaw
                self.camera_current_pitch = target_pitch
                self.camera_current_distance = target_distance
            current_pos = target_pos
            current_yaw = target_yaw
            current_pitch = target_pitch
            current_distance = target_distance

        # 计算视图矩阵
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=current_pos,
            distance=current_distance,
            yaw=current_yaw,
            pitch=current_pitch,
            roll=0,
            upAxisIndex=2
        )

        # 计算投影矩阵
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width / height,
            nearVal=0.1,
            farVal=100.0
        )

        # 获取相机图像
        (_, _, px, _, _) = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        if mode == "rgb_array":
            rgb_array = np.array(px).reshape(height, width, 4)
            rgb_array = rgb_array[:, :, :3]  # 去掉alpha通道
            return rgb_array

        return None

    def close(self):
        """
        关闭环境
        :return:
        """
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
