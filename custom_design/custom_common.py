import importlib.util

import numpy as np
import math
import os

from path_config import PROJECT_ROOT

CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
MAZE_SEARCH_DIR = os.path.join(CONFIG_DIR, 'maze_search')
PRIVATE_CONFIG_PATH = os.path.join(MAZE_SEARCH_DIR, 'curriculum_config.py')
PUBLIC_CONFIG_PATH = os.path.join(MAZE_SEARCH_DIR, 'train_single_config.py')


def load_config_function(config_path):
    """
    从指定的配置文件路径动态加载 get_reward_config 函数。
    Args:
        config_path (str): 配置文件路径（如 PRIVATE_CONFIG_PATH 或 PUBLIC_CONFIG_PATH）
    Returns:
        function: get_reward_config 函数，如果存在的话
    Raises:
        FileNotFoundError: 如果配置文件不存在
        AttributeError: 如果配置文件中没有 get_reward_config 函数
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    # 动态加载模块
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    if spec is None:
        raise ImportError(f"无法加载配置文件: {config_path}")

    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    # 获取 get_reward_config 函数
    if not hasattr(config_module, "get_reward_config"):
        raise AttributeError(f"配置文件 {config_path} 中未找到 get_reward_config 函数")

    return config_module.get_reward_config


def get_config(current_phase):
    """
    加载配置文件中的 get_reward_config 函数并获取配置，优先加载 private 配置文件。
    Args:
        current_phase: 当前阶段参数，传递给 get_reward_config 函数
    Returns:
        dict: 配置内容
    """
    # 优先尝试加载 private 配置文件
    try:
        if os.path.exists(PRIVATE_CONFIG_PATH):
            # print(f"加载完整配置文件: {PRIVATE_CONFIG_PATH}")
            get_reward_config_func = load_config_function(PRIVATE_CONFIG_PATH)
            return get_reward_config_func(current_phase)
    except (FileNotFoundError, AttributeError, ImportError) as e:
        print(f"加载完整配置文件失败: {e}")

    # 如果 private 配置文件加载失败，回退到 public 配置文件
    try:
        if os.path.exists(PUBLIC_CONFIG_PATH):
            # print(f"加载公开配置文件: {PUBLIC_CONFIG_PATH}")
            get_reward_config_func = load_config_function(PUBLIC_CONFIG_PATH)
            return get_reward_config_func(current_phase)
    except (FileNotFoundError, AttributeError, ImportError) as e:
        print(f"加载公开配置文件失败: {e}")
        raise FileNotFoundError(
            "未找到任何有效的配置文件！"
            "请确保 private/curriculum_config.py 或 public/train_single_config.py 存在"
            "并包含 get_reward_config 函数。")


def normalize_angle(angle):
    """将角度标准化到[-π, π]范围内"""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi
