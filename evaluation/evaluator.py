# -*- coding: utf-8 -*-
"""
@File    : evaluator.py
@Author  : zhangjian
@Desc    : 模型评估器
"""

import os
import time

import numpy as np
import pybullet as p
import traceback
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecVideoRecorder

from config.maze_search.curriculum_config import TRAINER_CONFIG
from evaluation.e_callbacks import PyBulletRenderWrapper


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


def get_base_env(env):
    """获取最底层的环境实例，穿透所有包装器"""
    # 处理向量化环境
    if hasattr(env, 'venv'):
        env = env.venv

    if hasattr(env, 'envs') and len(env.envs) > 0:
        env = env.envs[0]

    # 处理单环境包装器，如Monitor
    while hasattr(env, 'env'):
        env = env.env

    return env


def evaluate_sb3(model_path, env_class, curriculum_config, episodes=5, phase=2, render_mode="human", verbose=False):
    """
    评估训练好的SB3模型

    Args:
        model_path: 模型路径
        env_class: 环境类
        curriculum_config: 课程学习配置
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

    # 获取当前阶段的课程配置
    seed = curriculum_config["seed"]
    env_config = curriculum_config.get("env_config", {})
    phase = env_config["curriculum_phase"]
    # 创建环境
    env = env_class(maze_size=env_config["maze_size"], render_mode=render_mode, verbose=verbose, seed=seed)
    print(f"启动阶段-{phase}模型评估")

    # 优化视觉渲染以提高性能
    if hasattr(env, 'client_id'):
        print("优化PyBullet视觉渲染以提高性能")
        # 禁用阴影 - 阴影计算较为昂贵
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=env.client_id)
        # 减少GUI元素
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=env.client_id)
        # 禁用线框
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0, physicsClientId=env.client_id)
        # 降低点云渲染质量
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1, physicsClientId=env.client_id)

    # 根据阶段设置环境参数
    env.curriculum_phase = phase

    # 使用课程配置中的随机位置列表
    random_positions = env_config.get("random_positions", [])

    if random_positions:
        print(f"使用阶段{phase}的随机位置列表: {random_positions}")
    else:
        # 如果没有配置，使用默认位置
        if phase == 1:
            random_positions = [(1, 2), (2, 1)]
        elif phase == 2:
            random_positions = [(1, 3), (3, 1), (2, 3), (3, 2)]
        elif phase == 3:
            random_positions = [(1, 5), (5, 1), (4, 4), (5, 3), (3, 5)]
        print(f"使用默认随机位置列表: {random_positions}")

    # 设置环境的随机位置列表
    env.use_random_positions = True
    env.random_positions = random_positions

    # 包装环境以匹配训练时的结构
    env_m = Monitor(env)
    env = DummyVecEnv([lambda: env_m])
    # 在进入epoch之前先reset以确保正常显示球位置
    env.reset()
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
        time.sleep(1)
        # 评估统计
        episode_rewards = []
        episode_steps = []
        episode_successes = []  # 记录成功率

        for episode in range(episodes):
            # 在每个回合开始前随机选择球位置
            import random
            # ball_pos = random.choice(random_positions)

            print(f"\n===== 评估回合 {episode + 1}/{episodes} =====")

            # 获取最底层环境实例
            inner_env = get_base_env(env)
            print(f"上个回合球的位置：{inner_env.ball_pos}")
            # 直接设置内部环境的球位置
            # inner_env.ball_pos = ball_pos
            # inner_env.goal_pos = ball_pos

            # 先调用reset获取初始观察
            obs = env.reset()
            print(f"朝向角：{obs[0][2]}")
            # 重要：确保球位置设置正确反映在物理世界中
            if hasattr(inner_env, 'ball_id') and inner_env.ball_id is not None:
                world_pos, _ = p.getBasePositionAndOrientation(
                    inner_env.ball_id,
                    physicsClientId=inner_env.client_id
                )
                print(f"reset后球的网格位置: {inner_env.ball_pos}")
            else:
                print("警告：无法获取球体ID，无法设置物理位置")

            done = False
            total_reward = 0
            steps = 0
            manual_yaw = 0.0  # 初始化手动朝向
            time_step = 1.0 / 240.0  # 假设物理模拟时间步长为 1/240 秒

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)

                # 从向量环境中提取单个值
                reward = rewards[0]
                done = dones[0]
                info = infos[0]

                total_reward += reward
                steps += 1

                # 手动更新朝向：根据动作中的转向部分
                turn = action[0][1] * 0.25  # 与 _apply_action 中的转向缩放一致
                yaw_change = turn * time_step * 10.0  # 调整系数以匹配实际旋转
                manual_yaw += yaw_change
                manual_yaw = np.arctan2(np.sin(manual_yaw), np.cos(manual_yaw))

                # 计算并打印每个 step 的夹角调试信息
                try:
                    if (hasattr(inner_env, 'robot_id') and inner_env.robot_id is not None and
                            hasattr(inner_env, 'ball_id') and inner_env.ball_id is not None and
                            hasattr(inner_env, 'client_id') and inner_env.client_id is not None):
                        # 获取机器人的位置和朝向
                        robot_world_pos, robot_orientation = p.getBasePositionAndOrientation(
                            inner_env.robot_id,
                            physicsClientId=inner_env.client_id
                        )
                        robot_euler = p.getEulerFromQuaternion(robot_orientation)
                        pybullet_yaw = robot_euler[2]

                        # 使用手动计算的 yaw
                        robot_yaw = manual_yaw

                        # 获取小球的位置
                        ball_world_pos, _ = p.getBasePositionAndOrientation(
                            inner_env.ball_id,
                            physicsClientId=inner_env.client_id
                        )

                        # 计算机器人到小球的方向向量
                        dx = ball_world_pos[0] - robot_world_pos[0]
                        dy = ball_world_pos[1] - robot_world_pos[1]
                        target_angle = np.arctan2(dy, dx)

                        # 计算夹角（正面和背面假设）
                        angle_diff = target_angle - robot_yaw
                        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
                        angle_diff_deg = np.degrees(angle_diff)

                        angle_diff_back = target_angle - (robot_yaw + np.pi)
                        angle_diff_back = np.arctan2(np.sin(angle_diff_back), np.cos(angle_diff_back))
                        angle_diff_back_deg = np.degrees(angle_diff_back)

                        # 计算距离
                        calculated_distance = np.sqrt(dx ** 2 + dy ** 2)

                        # 每 20 步打印一次详细调试信息
                        if steps % 20 == 0 or done:
                            print(
                                f"步骤: {steps}, 调试夹角计算 - 机器人位置: {robot_world_pos}, 机器人四元数朝向: {robot_orientation}")
                            print(
                                f"步骤: {steps}, 调试夹角计算 - 机器人 Euler 角度 (roll, pitch, yaw): ({np.degrees(robot_euler[0]):.2f}, {np.degrees(robot_euler[1]):.2f}, {np.degrees(robot_euler[2]):.2f}) 度")
                            print(f"步骤: {steps}, 调试夹角计算 - 手动 yaw: {np.degrees(manual_yaw):.2f} 度")
                            print(
                                f"步骤: {steps}, 调试夹角计算 - 小球位置: {ball_world_pos}, 目标角度: {np.degrees(target_angle):.2f} 度")
                            print(
                                f"步骤: {steps}, 调试夹角计算 - 原始角度差 (正面): {np.degrees(target_angle - robot_yaw):.2f} 度, 归一化后: {angle_diff_deg:.2f} 度")
                            print(
                                f"步骤: {steps}, 调试夹角计算 - 原始角度差 (背面假设): {np.degrees(target_angle - (robot_yaw + np.pi)):.2f} 度, 归一化后: {angle_diff_back_deg:.2f} 度")
                            print(
                                f"步骤: {steps}, 调试夹角计算 - 计算的距离: {calculated_distance:.4f}, 日志中的距离: {info.get('distance_to_ball', 'N/A')}")
                    else:
                        print(f"步骤: {steps}, 无法获取机器人或小球ID，或缺少 client_id")
                except Exception as e:
                    print(f"步骤: {steps}, 夹角计算错误: {str(e)}")

                # 每20步打印一次状态
                if steps % 20 == 0 or done:
                    original_reward = info.get('original_reward', reward)
                    distance = info.get('distance_to_ball', 'N/A')
                    print(
                        f"步骤: {steps}, "
                        f"奖励: {original_reward:.2f}, "
                        f"累计奖励: {total_reward:.2f}, "
                        f"到球距离: {distance}，"
                        f"是否成功: {info.get('success', False)}",
                        f"是否超时: {info.get('timeout', False)}",
                        f"任务进度: {info.get('progress', 'N/A')}",
                    )

                    # 打印当前球位置，确认未被更改
                    if verbose and hasattr(inner_env, 'ball_id'):
                        world_pos, _ = p.getBasePositionAndOrientation(
                            inner_env.ball_id,
                            physicsClientId=inner_env.client_id
                        )
                        print(f"当前球物理位置: {world_pos}")

                time.sleep(0.1)

            # 记录本回合结果
            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            if isinstance(info, dict) and 'success' in info and info['success']:
                episode_successes.append(True)
            else:
                episode_successes.append(False)

            print(f"回合 {episode + 1} 结束，总步数: {steps}, 总奖励: {total_reward:.2f}")
            print(f"机器人正面朝向与小球的夹角: {angle_diff_deg:.1f} 度")
            print(f"机器人背面朝向与小球的夹角 (假设): {angle_diff_back_deg:.1f} 度")

        # 输出总体评估结果
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_steps = np.mean(episode_steps)
        success_rate = sum(episode_successes) / len(episode_successes) if episode_successes else 0.0

        print(f"\n===== 总体评估结果 =====")
        print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"平均步数: {mean_steps:.1f}")
        print(f"成功率: {success_rate * 100:.1f}% ({sum(episode_successes)}/{len(episode_successes)} 回合成功)")

        return mean_reward, std_reward

    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        traceback.print_exc()
        return None, None
    finally:
        # 确保关闭环境
        env.close()


def record_model_video(model_path, env_class, curriculum_config, episodes=1, phase=3,
                       video_length=1000, video_folder="videos", verbose=False):
    """
    录制模型执行视频

    Args:
        model_path: 模型路径
        env_class: 环境类
        curriculum_config: 课程学习配置
        episodes: 录制回合数
        phase: 课程阶段
        video_length: 每个视频的最大帧数
        video_folder: 视频保存文件夹
        verbose: 是否打印详细信息
    """
    print(f"开始录制模型视频: {model_path}")

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

    # 获取当前阶段的课程配置
    phase_config = curriculum_config["phases"].get(phase, {})
    env_config = phase_config.get("env_config", {})

    # 创建时间戳文件夹
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    video_folder = os.path.join(video_folder, f"{algorithm}_phase{phase}_{timestamp}")
    os.makedirs(video_folder, exist_ok=True)
    print(f"视频将保存到: {video_folder}")

    # 加载模型
    model = model_type.load(model_path)
    print(f"成功加载模型: {model_path}")

    # 为每个回合录制视频
    for episode in range(episodes):
        print(f"\n===== 录制回合 {episode + 1}/{episodes} =====")

        # 创建用于录制的环境 - 使用rgb_array渲染模式
        env = env_class(maze_size=(7, 7), render_mode="rgb_array", verbose=verbose)

        # 根据阶段设置环境参数
        env.curriculum_phase = phase

        # 使用课程配置中的随机位置列表
        random_positions = env_config.get("random_positions", [])
        if random_positions:
            print(f"使用阶段{phase}的随机位置列表")
        else:
            # 如果没有配置，使用默认位置
            if phase == 1:
                random_positions = [(1, 2), (2, 1)]
            elif phase == 2:
                random_positions = [(1, 3), (3, 1), (2, 3), (3, 2)]
            elif phase == 3:
                random_positions = [(1, 5), (5, 1), (4, 4), (5, 3), (3, 5)]
            print(f"使用默认随机位置列表")

        # 设置环境的随机位置列表
        env.use_random_positions = True
        env.random_positions = random_positions
        env_warp = PyBulletRenderWrapper(env, width=840, height=800)
        # 设置默认相机为俯视视角
        env_warp.set_camera("top_down")
        # 自定义相机参数
        env_warp.set_camera("follow", distance=10.0, yaw=60)
        env_warp.set_camera(pitch=-60)  # 调整为更高的视角
        # 创建向量环境
        vec_env = DummyVecEnv([lambda: env_warp])

        # 如果存在归一化文件，加载它
        if os.path.exists(vec_normalize_path):
            print(f"加载归一化统计数据: {vec_normalize_path}")
            vec_env = VecNormalize.load(vec_normalize_path, vec_env)
            # 在评估时禁用归一化更新
            vec_env.training = False
            vec_env.norm_reward = False

        # 设置视频录制器
        episode_video_path = os.path.join(video_folder, f"episode_{episode + 1}")
        vec_env = VecVideoRecorder(
            vec_env,
            episode_video_path,
            record_video_trigger=lambda x: x == 0,  # 在开始时触发录制
            video_length=video_length,
            name_prefix=f"{algorithm}_phase{phase}_ep{episode + 1}",
        )
        try:
            # 录制视频
            obs = vec_env.reset()
            total_reward = 0
            steps = 0

            for _ in range(video_length):
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = vec_env.step(action)

                # 从向量环境中提取单个值
                reward = rewards[0]
                done = dones[0]
                info = infos[0]

                total_reward += reward
                steps += 1

                # 每50步打印一次状态
                if verbose and steps % 50 == 0:
                    print(f"步骤: {steps}, 累计奖励: {total_reward:.2f}")

                # 如果回合结束，提前退出循环
                if done:
                    break

            print(f"回合 {episode + 1} 录制完成，总步数: {steps}, 总奖励: {total_reward:.2f}")

        finally:
            # 确保关闭环境和录制器
            vec_env.close()

    print(f"\n视频录制完成！视频保存在: {video_folder}")
