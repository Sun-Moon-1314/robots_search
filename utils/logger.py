# -*- coding: utf-8 -*-
"""
@File    : logger.py
@Author  : zhangjian
@Desc    : 日志工具
"""

import logging
import os
import time
from datetime import datetime


def setup_logger(name, log_dir="logs", level=logging.INFO):
    """
    设置日志记录器

    Args:
        name: 日志记录器名称
        log_dir: 日志目录
        level: 日志级别

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)

    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # 创建文件处理器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


class TrainingLogger:
    """训练日志记录器"""

    def __init__(self, name, log_dir="logs"):
        """
        初始化训练日志记录器

        Args:
            name: 日志名称
            log_dir: 日志目录
        """
        self.logger = setup_logger(name, log_dir)
        self.start_time = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_steps = 0

    def start_training(self):
        """记录训练开始"""
        self.start_time = time.time()
        self.logger.info("开始训练")

    def log_episode(self, episode, reward, length, info=None):
        """
        记录回合信息

        Args:
            episode: 回合编号
            reward: 回合奖励
            length: 回合长度
            info: 额外信息
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.total_steps += length

        # 计算平均奖励和长度
        avg_reward = sum(self.episode_rewards[-100:]) / min(100, len(self.episode_rewards))
        avg_length = sum(self.episode_lengths[-100:]) / min(100, len(self.episode_lengths))

        # 记录日志
        self.logger.info(f"回合 {episode} - 奖励: {reward:.2f}, 长度: {length}, "
                         f"平均奖励(100): {avg_reward:.2f}, 平均长度(100): {avg_length:.2f}")

        if info:
            self.logger.info(f"额外信息: {info}")

    def log_evaluation(self, mean_reward, std_reward, success_rate):
        """
        记录评估结果

        Args:
            mean_reward: 平均奖励
            std_reward: 奖励标准差
            success_rate: 成功率
        """
        self.logger.info(f"评估结果 - 平均奖励: {mean_reward:.2f} ± {std_reward:.2f}, "
                         f"成功率: {success_rate:.1f}%")

    def finish_training(self):
        """记录训练结束"""
        duration = time.time() - self.start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)

        self.logger.info(f"训练结束 - 总步数: {self.total_steps}, "
                         f"训练时间: {int(hours)}h {int(minutes)}m {int(seconds)}s")
