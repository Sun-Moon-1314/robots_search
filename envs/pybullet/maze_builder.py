# -*- coding: utf-8 -*-
"""
@File    : maze_builder.py
@Author  : zhangjian
@Desc    : 迷宫构建器
"""

import numpy as np
import pybullet as p
import pybullet_data


class MazeBuilder:
    """迷宫构建器，负责在PyBullet中构建迷宫环境"""

    def __init__(self, client_id, maze_size, cell_size=1.0, wall_height=0.5):
        """
        初始化迷宫构建器

        Args:
            client_id: PyBullet客户端ID
            maze_size: 迷宫大小，(width, height)元组
            cell_size: 单元格大小
            wall_height: 墙壁高度
        """
        self.client_id = client_id
        self.maze_size = maze_size
        self.cell_size = cell_size
        self.wall_height = wall_height
        self.wall_thickness = 0.1

        # 对象ID列表
        self.plane_id = None
        self.walls = []
        self.robot_id = None
        self.ball_id = None

    def generate_maze_data(self):
        """
        生成迷宫数据

        Returns:
            np.ndarray: 迷宫数据，0表示空地，1表示墙
        """
        # 创建一个固定的迷宫布局
        maze = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1]
        ])

        # 如果迷宫大小不是7x7，则调整迷宫大小
        if self.maze_size[0] != 7 or self.maze_size[1] != 7:
            # 创建一个更大或更小的迷宫
            new_maze = np.ones((self.maze_size[0], self.maze_size[1]), dtype=int)

            # 确保边界是墙
            new_maze[0, :] = 1
            new_maze[-1, :] = 1
            new_maze[:, 0] = 1
            new_maze[:, -1] = 1

            # 确保起点附近是空地
            new_maze[1, 1] = 0
            if self.maze_size[0] > 2 and self.maze_size[1] > 2:
                new_maze[1, 2] = 0
                new_maze[2, 1] = 0

            # 在中间添加一些墙和通道
            if self.maze_size[0] > 4:
                mid = self.maze_size[0] // 2
                new_maze[mid, 2:mid + 1] = 1  # 水平墙
                new_maze[2:mid + 1, mid] = 1  # 垂直墙
                new_maze[mid, mid] = 1  # 交叉点

            maze = new_maze

        return maze

    def build_ground(self):
        """构建地面"""
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        return self.plane_id

    def build_walls(self, maze_data):
        """
        构建墙壁

        Args:
            maze_data: 迷宫数据，0表示空地，1表示墙

        Returns:
            list: 墙壁ID列表
        """
        self.walls = []

        # 创建迷宫墙壁
        for i in range(self.maze_size[0]):
            for j in range(self.maze_size[1]):
                if maze_data[i][j] == 1:  # 墙壁
                    wall_x = i * self.cell_size + self.cell_size / 2
                    wall_y = j * self.cell_size + self.cell_size / 2

                    # 创建墙壁（使用立方体）
                    visual_shape_id = p.createVisualShape(
                        shapeType=p.GEOM_BOX,
                        halfExtents=[self.cell_size / 2, self.cell_size / 2, self.wall_height],
                        rgbaColor=[0.7, 0.7, 0.7, 1.0],
                        physicsClientId=self.client_id
                    )

                    collision_shape_id = p.createCollisionShape(
                        shapeType=p.GEOM_BOX,
                        halfExtents=[self.cell_size / 2, self.cell_size / 2, self.wall_height],
                        physicsClientId=self.client_id
                    )

                    wall_id = p.createMultiBody(
                        baseMass=0,  # 静态物体
                        baseCollisionShapeIndex=collision_shape_id,
                        baseVisualShapeIndex=visual_shape_id,
                        basePosition=[wall_x, wall_y, self.wall_height],
                        physicsClientId=self.client_id
                    )
                    self.walls.append(wall_id)

        return self.walls

    def place_robot(self, start_pos):
        """
        放置机器人

        Args:
            start_pos: 起始位置，(x, y)元组，网格坐标

        Returns:
            int: 机器人ID
        """
        start_x = start_pos[0] * self.cell_size + self.cell_size / 2
        start_y = start_pos[1] * self.cell_size + self.cell_size / 2

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot_id = p.loadURDF(
            "r2d2.urdf",
            [start_x, start_y, 0.5],  # 确保高度足够
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.client_id,
            useFixedBase=0,  # 确保不是固定的
            flags=p.URDF_USE_INERTIA_FROM_FILE  # 使用文件中的惯性参数
        )

        return self.robot_id

    def place_goal(self, goal_pos):
        """
        放置目标球

        Args:
            goal_pos: 目标位置，(x, y)元组，网格坐标

        Returns:
            int: 球ID
        """
        goal_x = goal_pos[1] * self.cell_size + self.cell_size / 2
        goal_y = goal_pos[0] * self.cell_size + self.cell_size / 2

        goal_visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.cell_size / 4,
            rgbaColor=[1.0, 0.0, 0.0, 1.0],  # 红色
            physicsClientId=self.client_id
        )

        goal_collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.cell_size / 4,
            physicsClientId=self.client_id
        )

        self.ball_id = p.createMultiBody(
            baseMass=1,  # 静态物体
            baseCollisionShapeIndex=goal_collision_shape_id,
            baseVisualShapeIndex=goal_visual_shape_id,
            basePosition=[goal_x, goal_y, 0.5],
            physicsClientId=self.client_id
        )

        return self.ball_id

    def reset_robot_position(self, start_pos):
        """
        重置机器人位置

        Args:
            start_pos: 起始位置，(x, y)元组，网格坐标
        """
        if self.robot_id is not None:
            start_x = start_pos[1] * self.cell_size + self.cell_size / 2
            start_y = start_pos[0] * self.cell_size + self.cell_size / 2
            p.resetBasePositionAndOrientation(
                self.robot_id,
                [start_x, start_y, 0.5],  # 确保高度足够
                p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.client_id
            )

            # 重置智能体速度
            p.resetBaseVelocity(
                self.robot_id,
                [0, 0, 0],  # 线速度
                [0, 0, 0],  # 角速度
                physicsClientId=self.client_id
            )

    def reset_goal_position(self, goal_pos):
        """
        重置目标位置

        Args:
            goal_pos: 目标位置，(x, y)元组，网格坐标
        """
        if self.ball_id is not None:
            goal_x = goal_pos[0] * self.cell_size + self.cell_size / 2
            goal_y = goal_pos[1] * self.cell_size + self.cell_size / 2
            p.resetBasePositionAndOrientation(
                self.ball_id,
                [goal_x, goal_y, 0.5],  # 位置
                [0, 0, 0, 1],  # 方向（四元数）
                physicsClientId=self.client_id
            )

    def is_valid_position(self, maze_data, pos_x, pos_y):
        """
        检查位置是否有效（不是墙壁且在迷宫内）

        Args:
            maze_data: 迷宫数据
            pos_x: x坐标
            pos_y: y坐标

        Returns:
            bool: 位置是否有效
        """
        # 检查该位置是否在迷宫内
        if pos_x < 0 or pos_x >= self.maze_size[0] or pos_y < 0 or pos_y >= self.maze_size[1]:
            return False

        # 检查该位置是否是墙壁
        if maze_data[pos_y][pos_x] == 1:  # 假设1表示墙壁
            return False

        return True

    def setup_camera(self, target_position=None):
        """
        设置相机视角

        Args:
            target_position: 目标位置，如果为None则使用迷宫中心
        """
        if target_position is None:
            # 计算迷宫中心
            maze_center_x = self.maze_size[0] * self.cell_size / 2
            maze_center_y = self.maze_size[1] * self.cell_size / 2
            target_position = [maze_center_x, maze_center_y, 0.0]

        # 初始化摄像机位置
        camera_distance = self.maze_size[0] * self.cell_size * 1.5
        camera_yaw = 45.0
        camera_pitch = -65.0

        # 设置初始摄像机位置
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=target_position,
            physicsClientId=self.client_id
        )

        # 确保启用鼠标控制
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
        # 确保渲染已启用
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

