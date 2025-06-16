# -*- coding: utf-8 -*-
"""
@File    : evaluator.py
@Author  : zhangjian
@Desc    : 模型评估器
"""

import os
import numpy as np
import time
import traceback
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


def evaluate_model(model, env, vec_normalize_path=None, n_eval_episodes=10):
    """
    评估模型性能

    Args:
        model: 训练好的模型
        env: 评估环境
        vec_normalize_path: 规范化环境路径
        n_eval_episodes: 评估回合数

    Returns:
        float: 平均奖励
    """
    # 如果提供了规范化环境路径，加载它
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        eval_env = VecNormalize.load(vec_normalize_path, env)
        # 在评估时禁用奖励规范化的更新，但仍使用已学习的统计数据
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = env

    episode_rewards = []
    for _ in range(n_eval_episodes):
        obs = eval_env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = eval_env.step(action)
            # 对于VecEnv，rewards是一个数组，我们只关心第一个环境
            reward = rewards[0] if isinstance(rewards, (list, np.ndarray)) else rewards
            episode_reward += reward
            # 对于VecEnv，dones是一个布尔数组
            done = dones[0] if isinstance(dones, (list, np.ndarray)) else dones
        episode_rewards.append(episode_reward)

    # 确保返回一个标量值
    mean_reward = np.mean(episode_rewards)
    return float(mean_reward)


def evaluate_sb3(model_path, env_class, episodes=5, phase=2, render_mode="human", verbose=False):
    """
    评估训练好的SB3模型

    Args:
        model_path: 模型路径
        env_class: 环境类
        episodes: 评估回合数
        phase: 课程阶段
        render_mode: 渲染模式
        verbose: 是否打印详细信息

    Returns:
        tuple: (平均奖励, 奖励标准差)
    """
    print(f"开始评估模型: {model_path}")

    # 确定模型类型
    if "PPO" in model_path:
        model_type = PPO
    elif "SAC" in model_path:
        model_type = SAC
    elif "A2C" in model_path:
        model_type = A2C
    else:
        raise ValueError(f"无法从路径确定算法类型: {model_path}")

    # 提取算法名称和阶段
    algorithm = os.path.basename(model_path).split("_")[0]
    model_phase = os.path.basename(model_path).split("_")[-1].split(".")[0]

    # 查找对应的归一化文件
    vec_normalize_path = os.path.join(os.path.dirname(model_path),
                                      f"vec_normalize_{algorithm}_{model_phase}.pkl")
    print(f"加载归一化文件:{vec_normalize_path}")
    # 创建环境
    env = env_class(maze_size=(7, 7), render_mode=render_mode, verbose=verbose)
    print(f"启动阶段-{phase}模型评估")

    # 根据阶段设置环境参数
    env.curriculum_phase = phase
    if phase == 1:
        env.ball_pos = (1, 2)
        env.goal_pos = env.ball_pos
    elif phase == 2:
        env.ball_pos = (1, 3)
        env.goal_pos = env.ball_pos
    # phase 3使用默认设置
    elif phase == 3:
        env.ball_pos = (1, 3)
        env.goal_pos = env.ball_pos
    # 包装环境以匹配训练时的结构
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # 如果存在归一化文件，加载它
    if os.path.exists(vec_normalize_path):
        print(f"加载归一化统计数据: {vec_normalize_path}")
        env = VecNormalize.load(vec_normalize_path, env)
        # 在评估时禁用归一化更新
        env.training = False
        env.norm_reward = False

    try:
        # 加载模型
        model = model_type.load(model_path)
        print(f"成功加载模型: {model_path}")

        # 评估统计
        episode_rewards = []
        episode_steps = []

        for episode in range(episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0

            print(f"\n===== 评估回合 {episode + 1}/{episodes} =====")

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)

                # 从向量环境中提取单个值
                reward = rewards[0]
                done = dones[0]
                info = infos[0]

                total_reward += reward
                steps += 1

                # 每20步打印一次状态
                if steps % 20 == 0:
                    # 提取原始奖励(如果可用)
                    original_reward = info.get('original_reward', reward)
                    distance = info.get('distance_to_ball', 'N/A')
                    print(
                        f"步骤: {steps}, 奖励: {original_reward:.2f}, 累计奖励: {total_reward:.2f}, 到球距离: {distance}")

            # 记录本回合结果
            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            print(f"回合 {episode + 1} 结束，总步数: {steps}, 总奖励: {total_reward:.2f}")

        # 输出总体评估结果
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_steps = np.mean(episode_steps)

        print(f"\n===== 总体评估结果 =====")
        print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"平均步数: {mean_steps:.1f}")
        print(f"成功率: {sum(r > 0 for r in episode_rewards) / episodes * 100:.1f}%")

        return mean_reward, std_reward

    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        traceback.print_exc()
        return None, None
    finally:
        # 确保关闭环境
        env.close()
