import numpy as np
from gymnasium import Wrapper


class PyBulletRenderWrapper(Wrapper):
    """PyBullet环境的渲染包装器，提供灵活的相机控制"""

    def __init__(self, env, width=320, height=240):
        """
        初始化渲染包装器

        参数:
            env: 被包装的环境
            width: 渲染图像宽度
            height: 渲染图像高度
        """
        super().__init__(env)
        self._width = width
        self._height = height

        # 默认相机配置
        self._camera_configs = {
            "top_down": {
                "distance": 10.0,
                "yaw": 0,
                "pitch": -89,
                "target_position_func": self._get_center_position
            },
            "follow": {
                "distance": 3.0,
                "yaw": 45,
                "pitch": -30,
                "target_position_func": self._get_robot_position
            },
            "isometric": {
                "distance": 7.0,
                "yaw": 45,
                "pitch": -45,
                "target_position_func": self._get_robot_position
            }
        }

        # 默认使用的相机类型
        self._default_camera = "follow"

    def _get_robot_position(self):
        """获取机器人位置作为相机目标点"""
        import pybullet as p

        # 尝试从环境中获取机器人ID
        robot_id = None
        if hasattr(self.env, 'robot_id'):
            robot_id = self.env.robot_id

        # 如果找到机器人ID，返回其位置
        if robot_id is not None:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            return [pos[0], pos[1], 0]  # 只使用x,y坐标，z设为0

        # 如果没有找到机器人，返回默认位置
        return [0, 0, 0]

    def _get_center_position(self):
        """获取环境中心位置作为相机目标点"""
        # 尝试从环境中获取迷宫尺寸和中心
        if hasattr(self.env, 'maze_size'):
            maze_size = self.env.maze_size
            return [maze_size[0] / 2, maze_size[1] / 2, 0]

        # 如果没有迷宫尺寸信息，尝试使用机器人位置
        return self._get_robot_position()

    def set_camera(self, camera_type=None, **kwargs):
        """
        设置相机参数

        参数:
            camera_type: 相机类型，可选 "top_down", "follow", "isometric" 或 None
            **kwargs: 其他相机参数，可包括 distance, yaw, pitch, target_position
        """
        if camera_type is not None:
            self._default_camera = camera_type

        # 如果提供了特定参数，更新相机配置
        if kwargs and self._default_camera in self._camera_configs:
            for key, value in kwargs.items():
                if key in self._camera_configs[self._default_camera]:
                    self._camera_configs[self._default_camera][key] = value

    def render(self, mode='rgb_array', camera_type=None):
        """
        渲染环境

        参数:
            mode: 渲染模式，'rgb_array'返回图像数组
            camera_type: 相机类型，如果为None则使用默认相机

        返回:
            根据mode返回相应的渲染结果
        """
        if mode != 'rgb_array':
            return self.env.render(mode=mode)

        import pybullet as p

        # 确定使用的相机类型
        cam_type = camera_type if camera_type is not None else self._default_camera
        if cam_type not in self._camera_configs:
            cam_type = next(iter(self._camera_configs))  # 使用第一个可用的相机类型

        # 获取相机配置
        config = self._camera_configs[cam_type]

        # 获取目标位置
        if "target_position" in config:
            target_position = config["target_position"]
        elif "target_position_func" in config and callable(config["target_position_func"]):
            target_position = config["target_position_func"]()
        else:
            target_position = [0, 0, 0]

        # 计算视图矩阵
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target_position,
            distance=config.get("distance", 5.0),
            yaw=config.get("yaw", 0),
            pitch=config.get("pitch", -30),
            roll=config.get("roll", 0),
            upAxisIndex=2
        )

        # 计算投影矩阵
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=config.get("fov", 60),
            aspect=float(self._width) / self._height,
            nearVal=config.get("nearVal", 0.1),
            farVal=config.get("farVal", 100.0)
        )

        # 获取相机图像
        (_, _, px, _, _) = p.getCameraImage(
            width=self._width,
            height=self._height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # 转换为RGB数组
        rgb_array = np.array(px).reshape(self._height, self._width, 4)
        rgb_array = rgb_array[:, :, :3]  # 去掉alpha通道

        return rgb_array

    def set_render_size(self, width, height):
        """设置渲染图像的尺寸"""
        self._width = width
        self._height = height
