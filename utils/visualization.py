# -*- coding: utf-8 -*-
"""
@File    : visualization.py
@Author  : zhangjian
@Desc    : 可视化工具
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_results(rewards, lengths, title="训练进度", save_path=None):
    """
    绘制训练结果图表

    Args:
        rewards: 奖励列表
        lengths: 回合长度列表
        title: 图表标题
        save_path: 保存路径，如果为None则显示图表
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 绘制奖励曲线
    ax1.plot(rewards, label='回合奖励')
    ax1.plot(pd.Series(rewards).rolling(100).mean(), label='平均奖励(100回合)', color='orange')
    ax1.set_ylabel('奖励')
    ax1.set_title(f'{title} - 奖励')
    ax1.legend()
    ax1.grid(True)

    # 绘制长度曲线
    ax2.plot(lengths, label='回合长度')
    ax2.plot(pd.Series(lengths).rolling(100).mean(), label='平均长度(100回合)', color='orange')
    ax2.set_xlabel('回合')
    ax2.set_ylabel('步数')
    ax2.set_title(f'{title} - 回合长度')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_evaluation_results(rewards, success_rates, algorithms=None, title="算法比较", save_path=None):
    """
    绘制评估结果比较图表

    Args:
        rewards: 各算法的奖励列表
        success_rates: 各算法的成功率列表
        algorithms: 算法名称列表
        title: 图表标题
        save_path: 保存路径，如果为None则显示图表
    """
    if algorithms is None:
        algorithms = [f"算法{i + 1}" for i in range(len(rewards))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 绘制奖励对比
    ax1.bar(algorithms, [np.mean(r) for r in rewards], yerr=[np.std(r) for r in rewards])
    ax1.set_ylabel('平均奖励')
    ax1.set_title(f'{title} - 平均奖励')
    ax1.grid(True, axis='y')

    # 绘制成功率对比
    ax2.bar(algorithms, success_rates)
    ax2.set_ylabel('成功率 (%)')
    ax2.set_title(f'{title} - 成功率')
    ax2.grid(True, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_curriculum_progress(phase_rewards, phase_names=None, title="课程学习进度", save_path=None):
    """
    绘制课程学习进度图表

    Args:
        phase_rewards: 各阶段的奖励列表字典
        phase_names: 阶段名称列表
        title: 图表标题
        save_path: 保存路径，如果为None则显示图表
    """
    if phase_names is None:
        phase_names = [f"阶段{i + 1}" for i in range(len(phase_rewards))]

    fig, ax = plt.subplots(figsize=(12, 6))

    # 为每个阶段绘制奖励曲线
    for i, (phase, rewards) in enumerate(phase_rewards.items()):
        episodes = range(len(rewards))
        ax.plot(episodes, rewards, label=phase_names[i])

    ax.set_xlabel('训练步数')
    ax.set_ylabel('平均奖励')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
