# -*- coding: utf-8 -*-
"""
@File    : curriculum.py
@Author  : zhangjian
@Desc    : 课程学习实现
"""

import os
import random
from copy import deepcopy
import copy
import numpy as np
import pickle
import time
import datetime
from stable_baselines3 import PPO, SAC, HerReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config.maze_search.default_config import MODELS_DIR, LOGS_DIR
from evaluation.evaluator import evaluate_model
from training.curriculum_callbacks import CurriculumEvalCallback, SaveLatestModelCallback, \
    SaveCheckpointCallback, EarlyStoppingException, StepwiseLRSchedulerCallback

# 创建保存最佳模型的目录
best_model_dir = os.path.join(MODELS_DIR, f"best_models")
os.makedirs(best_model_dir, exist_ok=True)


def create_env_from_config(env_class, env_config, is_eval=False):
    """
    根据配置创建环境

    Args:
        env_class: 环境类
        env_config: 环境配置字典
        is_eval: 是否为评估环境

    Returns:
        创建好的环境
    """
    # 复制配置以避免修改原始配置
    env_config = deepcopy(env_config)

    # 创建基础环境
    env = env_class(maze_size=env_config.get("maze_size", (7, 7)))

    # 设置课程阶段
    env.curriculum_phase = env_config.get("curriculum_phase", 1)

    # 处理随机位置
    if env_config.get("random_positions", False):
        # 环境内部随机生成位置
        pass
    else:
        # 设置固定位置
        if "ball_pos" in env_config:
            env.ball_pos = env_config["ball_pos"]

        # 处理特殊值
        if "goal_pos" in env_config:
            if env_config["goal_pos"] == "ball_pos":
                env.goal_pos = env.ball_pos
            else:
                env.goal_pos = env_config["goal_pos"]

    # 包装环境
    env = Monitor(env)  # 返回真实奖励
    env = DummyVecEnv([lambda: env])

    # 关键修改：确保训练和评估使用相同的归一化统计数据
    if is_eval:
        # 评估环境：使用训练环境的统计数据，但不更新
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.,
            clip_reward=10.,
            gamma=0.99,
            training=False  # 评估时不更新统计数据
        )
    else:
        # 训练环境：正常创建
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.,
            clip_reward=10.,
            gamma=0.99,
            training=True
        )

    return env


def create_model_from_config(algorithm, env, model_params):
    """
    根据配置创建模型

    Args:
        algorithm: 算法名称 ("SAC" 或 "PPO")
        env: 训练环境
        model_params: 模型参数字典

    Returns:
        创建好的模型
    """
    if algorithm == "SAC":
        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=LOGS_DIR, **model_params)
    elif algorithm == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOGS_DIR, **model_params)
    else:
        raise ValueError(f"不支持的算法: {algorithm}")

    return model


def save_training_state(algorithm, phase_complete, phase, timesteps_per_phase, model=None, env=None, total_steps=0,
                        attempt=0, phase_config=None):
    """
    保存训练状态到文件，包含更多训练参数并保存模型、环境和经验回放池

    Args:
        algorithm: 算法名称
        phase_complete: 各阶段完成状态字典
        phase: 当前训练阶段
        timesteps_per_phase: 当前阶段的总训练步数设置
        model: 当前训练模型 (可选)
        env: 当前训练环境 (可选)
        total_steps: 当前阶段已完成的步数 (可选)
        attempt: 当前阶段的尝试次数 (可选)
        phase_config: 当前阶段的配置 (可选)
    """
    state_dir = os.path.join(MODELS_DIR, "states")
    os.makedirs(state_dir, exist_ok=True)

    # 基本训练状态
    state = {
        "phase_complete": phase_complete,
        "current_phase": phase,
        "timesteps_per_phase": timesteps_per_phase,
        "total_steps_in_phase": total_steps,
        "attempt": attempt,  # 添加尝试次数
        "save_time": {
            "timestamp": time.time(),
            "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    # 添加模型参数
    if model is not None:
        # 获取模型的学习率
        if hasattr(model, "learning_rate"):
            state["learning_rate"] = model.learning_rate
        elif hasattr(model, "lr_schedule") and callable(model.lr_schedule):
            state["learning_rate"] = model.lr_schedule(total_steps)

        # 获取其他训练参数
        if algorithm == "PPO" and hasattr(model, "clip_range"):
            state["clip_range"] = model.clip_range
            if hasattr(model, "ent_coef"):
                state["ent_coef"] = model.ent_coef
            if hasattr(model, "vf_coef"):
                state["vf_coef"] = model.vf_coef
            if hasattr(model, "gamma"):
                state["gamma"] = model.gamma
            if hasattr(model, "gae_lambda"):
                state["gae_lambda"] = model.gae_lambda
            if hasattr(model, "n_steps"):
                state["n_steps"] = model.n_steps
            if hasattr(model, "n_epochs"):
                state["n_epochs"] = model.n_epochs
        elif algorithm == "SAC":
            if hasattr(model, "gamma"):
                state["gamma"] = model.gamma
            if hasattr(model, "tau"):
                state["tau"] = model.tau
            if hasattr(model, "buffer_size"):
                state["buffer_size"] = model.buffer_size
            if hasattr(model, "batch_size"):
                state["batch_size"] = model.batch_size
            if hasattr(model, "ent_coef"):
                state["ent_coef"] = model.ent_coef

        # 保存模型
        model_path = os.path.join(state_dir, f"{algorithm}_phase{phase}_state_model")
        model.save(model_path)
        state["model_path"] = model_path
        print(f"模型已保存: {model_path}.zip")

        # 保存经验回放池（如果模型有）
        if hasattr(model, "replay_buffer"):
            replay_buffer_path = os.path.join(state_dir, f"{algorithm}_phase{phase}_state_replay_buffer.pkl")
            try:
                model.save_replay_buffer(replay_buffer_path)
                state["replay_buffer_path"] = replay_buffer_path
                print(f"经验回放池已保存: {replay_buffer_path}")
            except Exception as e:
                print(f"保存经验回放池失败: {e}")

    # 添加环境参数并保存环境状态
    if env is not None:
        if hasattr(env, "norm_reward") and hasattr(env, "norm_obs"):
            state["env_normalization"] = {
                "norm_obs": env.norm_obs,
                "norm_reward": env.norm_reward
            }
            if hasattr(env, "clip_obs"):
                state["env_normalization"]["clip_obs"] = env.clip_obs
            if hasattr(env, "clip_reward"):
                state["env_normalization"]["clip_reward"] = env.clip_reward
            if hasattr(env, "gamma"):
                state["env_normalization"]["gamma"] = env.gamma

        # 保存环境状态（如果是VecNormalize环境）
        if hasattr(env, "save"):
            env_path = os.path.join(state_dir, f"{algorithm}_phase{phase}_state_env.pkl")
            try:
                env.save(env_path)
                state["env_state_path"] = env_path
                print(f"环境状态已保存: {env_path}")
            except Exception as e:
                print(f"保存环境状态失败: {e}")

    # 保存随机数生成器状态
    random_state = {
        "numpy": np.random.get_state(),
        "python": random.getstate()
    }

    # 如果使用PyTorch，也保存其随机数状态
    try:
        import torch
        random_state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            random_state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    except Exception as e:
        print(f"保存PyTorch随机数状态失败: {e}")

    state["random_state"] = random_state

    # 添加阶段配置信息
    if phase_config is not None:
        state["phase_config"] = {
            "name": phase_config.get("name", f"Phase {phase}"),
            "reward_threshold": phase_config.get("reward_threshold", 0),
            "env_config": phase_config.get("env_config", {}),
            "max_attempts": phase_config.get("max_attempts", 3),  # 添加最大尝试次数
            "step_increase_factor": phase_config.get("step_increase_factor", 1.5)  # 添加步数增加因子
        }
        # 添加模型参数，但排除不可序列化的对象
        if "model_params" in phase_config:
            model_params = {}
            for algo, params in phase_config["model_params"].items():
                model_params[algo] = {}
                for key, value in params.items():
                    if isinstance(value, (int, float, str, bool, list, dict, tuple)) or value is None:
                        model_params[algo][key] = value
            state["phase_config"]["model_params"] = model_params

    state_path = os.path.join(state_dir, f"{algorithm}_training_state.pkl")
    with open(state_path, "wb") as f:
        pickle.dump(state, f)
    print(f"训练状态已保存: {state_path}")
    return state


def load_training_state(algorithm, total_phases):
    """从文件加载训练状态"""
    state_path = os.path.join(MODELS_DIR, "states", f"{algorithm}_training_state.pkl")
    if os.path.exists(state_path):
        try:
            with open(state_path, "rb") as f:
                state = pickle.load(f)

            # 打印加载的状态信息
            print(f"已加载训练状态: {state_path}")
            if "save_time" in state:
                print(f"  - 保存时间: {state['save_time'].get('datetime', '未知')}")
            print(f"  - 当前阶段: {state.get('current_phase', '未知')}")

            if "learning_rate" in state:
                print(f"  - 学习率: {state['learning_rate']}")

            if "total_steps_in_phase" in state:
                print(f"  - 当前阶段已完成步数: {state['total_steps_in_phase']}")

            phase_complete = state.get("phase_complete", {})
            for phase, completed in sorted(phase_complete.items()):
                status = "已完成" if completed else "未完成"
                print(f"  - 阶段{phase}: {status}")

            # 检查模型和环境文件是否存在
            if "model_path" in state:
                model_file = f"{state['model_path']}.zip"
                if os.path.exists(model_file):
                    print(f"  - 模型文件: {model_file} (存在)")
                else:
                    print(f"  - 模型文件: {model_file} (不存在)")

            if "env_state_path" in state:
                if os.path.exists(state["env_state_path"]):
                    print(f"  - 环境状态文件: {state['env_state_path']} (存在)")
                else:
                    print(f"  - 环境状态文件: {state['env_state_path']} (不存在)")

            if "replay_buffer_path" in state:
                if os.path.exists(state["replay_buffer_path"]):
                    print(f"  - 经验回放池文件: {state['replay_buffer_path']} (存在)")
                else:
                    print(f"  - 经验回放池文件: {state['replay_buffer_path']} (不存在)")

            return state
        except Exception as e:
            print(f"加载训练状态失败: {e}")
            return {
                "phase_complete": {i: False for i in range(1, total_phases + 1)},
                "current_phase": 1,
                "timesteps_per_phase": None,
                "total_steps_in_phase": 0,
                "attempt": 0
            }
    else:
        print(f"未找到训练状态文件: {state_path}")
        return {
            "phase_complete": {i: False for i in range(1, total_phases + 1)},
            "current_phase": 1,
            "timesteps_per_phase": None,
            "total_steps_in_phase": 0,
            "attempt": 0
        }


def train_with_curriculum(env_class, curriculum_config, resume=False, phase_set=None):
    """
    使用课程学习训练智能体

    Args:
        env_class: 环境类
        curriculum_config: 课程学习配置字典
        resume: 是否从已保存的模型继续训练
        phase_set: 强制设置某个阶段为完成并从下一阶段开始

    Returns:
        训练结果字典
    """
    # 获取配置
    algorithm = curriculum_config["algorithm"]
    total_phases = curriculum_config["total_phases"]
    base_timesteps_per_phase = curriculum_config["timesteps_per_phase"]
    eval_freq = curriculum_config["eval_freq"]

    # 初始化或加载训练状态
    if resume:
        state = load_training_state(algorithm, total_phases)
        phase_complete = state["phase_complete"]
        start_phase = state["current_phase"]

        # 处理phase_set参数（强制设置某个阶段为完成并从下一阶段开始）
        if phase_set is not None:
            # 将指定阶段标记为已完成
            phase_complete[phase_set] = True
            # 从下一个阶段开始训练
            start_phase = phase_set + 1
            print(f"已将阶段{phase_set}标记为完成，从阶段{start_phase}开始训练")
        # 常规恢复训练逻辑
        else:
            # 检查是否所有阶段都已完成
            all_complete = all(phase_complete.values())
            if all_complete:
                print("课程学习已完成")
                return {f"phase{i}_complete": phase_complete[i] for i in range(1, total_phases + 1)}

            # 如果当前阶段已完成，进入下一阶段
            if phase_complete[start_phase]:
                start_phase += 1

        print(f"从阶段{start_phase}继续训练")
        # 确保start_phase不超过总阶段数
        if start_phase > total_phases:
            print(f"所有阶段已完成，无需继续训练")
            return {f"phase{i}_complete": phase_complete[i] for i in range(1, total_phases + 1)}

        # 恢复上次保存的timesteps_per_phase（如果有）
        saved_timesteps = state.get("timesteps_per_phase")
        if saved_timesteps is not None:
            base_timesteps_per_phase = saved_timesteps
            print(f"恢复训练步数设置: {base_timesteps_per_phase}")

        # 恢复上次保存的total_steps（如果有）
        saved_total_steps = state.get("total_steps_in_phase", 0)
        if saved_total_steps > 0:
            print(f"恢复已完成步数: {saved_total_steps}")

        # 恢复随机数生成器状态
        if "random_state" in state:
            try:
                np.random.set_state(state["random_state"]["numpy"])
                random.setstate(state["random_state"]["python"])

                # 恢复PyTorch随机数状态（如果有）
                try:
                    import torch
                    if "torch" in state["random_state"]:
                        torch.set_rng_state(state["random_state"]["torch"])
                    if "torch_cuda" in state["random_state"] and torch.cuda.is_available():
                        if isinstance(state["random_state"]["torch_cuda"], list):
                            torch.cuda.set_rng_state_all(state["random_state"]["torch_cuda"])
                        else:
                            torch.cuda.set_rng_state(state["random_state"]["torch_cuda"])
                    print("已恢复PyTorch随机数生成器状态")
                except (ImportError, KeyError, RuntimeError) as e:
                    print(f"恢复PyTorch随机数状态失败: {e}")

                print("已恢复随机数生成器状态")
            except Exception as e:
                print(f"恢复随机数生成器状态失败: {e}")
    else:
        # 全新训练
        phase_complete = {i: False for i in range(1, total_phases + 1)}
        start_phase = 1
        print("开始新的课程学习训练")

    # 逐阶段训练
    for phase in range(start_phase, total_phases + 1):
        phase_config = curriculum_config["phases"][phase]
        print(f"阶段{phase}: {phase_config['name']}...")

        # 创建训练环境
        env = create_env_from_config(env_class, phase_config["env_config"], is_eval=False)

        # 创建评估环境
        eval_env = create_env_from_config(env_class, phase_config["env_config"], is_eval=True)

        # 获取当前阶段的模型参数
        model_params = phase_config["model_params"][algorithm]

        # 创建模型
        model = create_model_from_config(algorithm, env, model_params)

        # 处理模型加载逻辑
        if resume and phase == start_phase:
            # 恢复训练：优先加载保存的训练状态中的模型
            if "model_path" in state and os.path.exists(f"{state['model_path']}.zip"):
                print(f"从保存的训练状态加载模型: {state['model_path']}")
                try:
                    # 使用load方法加载模型，但保留当前阶段的学习率设置
                    model = type(model).load(state["model_path"], env=env)

                    # 更新模型的学习率为当前阶段配置的学习率
                    if hasattr(model, "learning_rate"):
                        model.learning_rate = model_params["learning_rate"]
                        print(f"已更新学习率为当前阶段设置: {model_params['learning_rate']}")

                    print("模型加载成功")

                    # 恢复环境状态（如果有）
                    if "env_state_path" in state and state["env_state_path"] and os.path.exists(
                            state["env_state_path"]):
                        try:
                            from stable_baselines3.common.vec_env import VecNormalize
                            if isinstance(env, VecNormalize):
                                env = VecNormalize.load(state["env_state_path"], env)
                                print(f"已加载环境归一化状态: {state['env_state_path']}")
                            elif hasattr(env, "load"):
                                env.load(state["env_state_path"])
                                print(f"已加载环境状态: {state['env_state_path']}")
                        except Exception as e:
                            print(f"加载环境状态失败: {e}")

                    # 恢复经验回放池（如果有）
                    if "replay_buffer_path" in state and state["replay_buffer_path"] and os.path.exists(
                            state["replay_buffer_path"]):
                        try:
                            model.load_replay_buffer(state["replay_buffer_path"])
                            print(f"已加载经验回放池: {state['replay_buffer_path']}")
                        except Exception as e:
                            print(f"加载经验回放池失败: {e}")
                except Exception as e:
                    print(f"从训练状态加载模型失败: {e}，尝试其他模型加载方式")
                    # 如果加载失败，继续尝试其他加载方式

            # 如果没有找到训练状态中的模型或加载失败，则按原来的逻辑尝试加载
            if not ("model_path" in state and os.path.exists(f"{state['model_path']}.zip")):
                latest_path = os.path.join(MODELS_DIR, f"{algorithm}_phase{phase}_latest")
                best_path = os.path.join(MODELS_DIR, f"{algorithm}_phase{phase}_best")
                phase_path = os.path.join(MODELS_DIR, f"{algorithm}_phase{phase}")

                # 处理强制跳过阶段的情况
                if phase_set is not None and phase == start_phase:
                    # 尝试加载前一阶段的模型作为起点
                    prev_phase_path = os.path.join(MODELS_DIR, f"{algorithm}_phase{phase_set}")
                    prev_best_path = os.path.join(MODELS_DIR, f"{algorithm}_phase{phase_set}_best")
                    prev_latest_path = os.path.join(MODELS_DIR, f"{algorithm}_phase{phase_set}_latest")

                    # 按优先级尝试加载前一阶段的各种模型
                    model_loaded = False
                    for path_name, path in [
                        ("最佳模型", prev_best_path),
                        ("最新模型", prev_latest_path),
                        ("完成模型", prev_phase_path)
                    ]:
                        if os.path.exists(f"{path}.zip"):
                            print(f"强制跳过：从阶段{phase_set}的{path_name}加载")
                            model = type(model).load(path, env=env)

                            # 更新模型的学习率为当前阶段配置的学习率
                            if hasattr(model, "learning_rate"):
                                model.learning_rate = model_params["learning_rate"]
                                print(f"已更新学习率为当前阶段设置: {model_params['learning_rate']}")

                            # 尝试加载对应的环境归一化状态
                            vec_norm_path = f"{path}_vecnorm.pkl"
                            if os.path.exists(vec_norm_path):
                                try:
                                    from stable_baselines3.common.vec_env import VecNormalize
                                    env = VecNormalize.load(vec_norm_path, env)
                                    print(f"已加载环境归一化状态: {vec_norm_path}")
                                except Exception as e:
                                    print(f"加载环境归一化状态失败: {e}")
                            model_loaded = True
                            break

                    # 如果没有找到模型，尝试检查点
                    if not model_loaded:
                        # 查找前一阶段的最新检查点
                        prev_checkpoints = [f for f in os.listdir(MODELS_DIR)
                                            if f.startswith(f"{algorithm}_phase{phase_set}_checkpoint")
                                            and f.endswith(".zip")]
                        if prev_checkpoints:
                            # 按修改时间排序，选择最新的
                            latest_checkpoint = sorted(prev_checkpoints,
                                                       key=lambda x: os.path.getmtime(os.path.join(MODELS_DIR, x)))[-1]
                            checkpoint_path = os.path.join(MODELS_DIR, latest_checkpoint.replace(".zip", ""))
                            print(f"强制跳过：从阶段{phase_set}的检查点模型加载: {latest_checkpoint}")
                            model = type(model).load(checkpoint_path, env=env)

                            # 更新模型的学习率为当前阶段配置的学习率
                            if hasattr(model, "learning_rate"):
                                model.learning_rate = model_params["learning_rate"]
                                print(f"已更新学习率为当前阶段设置: {model_params['learning_rate']}")

                            # 尝试加载对应的环境归一化状态
                            vec_norm_path = f"{checkpoint_path}_vecnorm.pkl"
                            if os.path.exists(vec_norm_path):
                                try:
                                    from stable_baselines3.common.vec_env import VecNormalize
                                    env = VecNormalize.load(vec_norm_path, env)
                                    print(f"已加载环境归一化状态: {vec_norm_path}")
                                except Exception as e:
                                    print(f"加载环境归一化状态失败: {e}")
                        else:
                            print(f"警告：未找到阶段{phase_set}的任何模型，使用新初始化的模型")

                    # 尝试加载前一阶段的回放经验池
                    prev_buffer_path = os.path.join(MODELS_DIR, f"{algorithm}_phase{phase_set}_replay_buffer.pkl")
                    if os.path.exists(prev_buffer_path) and hasattr(model, "replay_buffer"):
                        try:
                            model.load_replay_buffer(prev_buffer_path)
                            print(f"已加载阶段{phase_set}的回放经验池: {prev_buffer_path}")
                        except Exception as e:
                            print(f"加载回放经验池失败: {e}")
                else:
                    # 正常情况下尝试加载当前阶段的模型
                    model_loaded = False
                    for path_name, path in [
                        ("最新模型", latest_path),
                        ("最佳模型", best_path),
                        ("完成模型", phase_path)
                    ]:
                        if os.path.exists(f"{path}.zip"):
                            print(f"恢复训练：加载阶段{phase}的{path_name}")
                            model = type(model).load(path, env=env)

                            # 更新模型的学习率为当前阶段配置的学习率
                            if hasattr(model, "learning_rate"):
                                model.learning_rate = model_params["learning_rate"]
                                print(f"已更新学习率为当前阶段设置: {model_params['learning_rate']}")

                            # 尝试加载对应的环境归一化状态
                            vec_norm_path = f"{path}_vecnorm.pkl"
                            if os.path.exists(vec_norm_path):
                                try:
                                    from stable_baselines3.common.vec_env import VecNormalize
                                    env = VecNormalize.load(vec_norm_path, env)
                                    print(f"已加载环境归一化状态: {vec_norm_path}")
                                except Exception as e:
                                    print(f"加载环境归一化状态失败: {e}")
                            model_loaded = True
                            break

                    # 如果没有找到模型，尝试检查点
                    if not model_loaded:
                        # 查找最新的检查点
                        checkpoints = [f for f in os.listdir(MODELS_DIR)
                                       if f.startswith(f"{algorithm}_phase{phase}_checkpoint")
                                       and f.endswith(".zip")]
                        if checkpoints:
                            # 按修改时间排序，选择最新的
                            latest_checkpoint = sorted(checkpoints,
                                                       key=lambda x: os.path.getmtime(os.path.join(MODELS_DIR, x)))[-1]
                            checkpoint_path = os.path.join(MODELS_DIR, latest_checkpoint.replace(".zip", ""))
                            print(f"恢复训练：加载阶段{phase}的检查点模型: {latest_checkpoint}")
                            model = type(model).load(checkpoint_path, env=env)

                            # 更新模型的学习率为当前阶段配置的学习率
                            if hasattr(model, "learning_rate"):
                                model.learning_rate = model_params["learning_rate"]
                                print(f"已更新学习率为当前阶段设置: {model_params['learning_rate']}")

                            # 尝试加载对应的环境归一化状态
                            vec_norm_path = f"{checkpoint_path}_vecnorm.pkl"
                            if os.path.exists(vec_norm_path):
                                try:
                                    from stable_baselines3.common.vec_env import VecNormalize
                                    env = VecNormalize.load(vec_norm_path, env)
                                    print(f"已加载环境归一化状态: {vec_norm_path}")
                                except Exception as e:
                                    print(f"加载环境归一化状态失败: {e}")
                        else:
                            print(f"警告：未找到阶段{phase}的任何模型，使用新初始化的模型")

                    # 正常情况下尝试加载上一阶段的回放经验池
                    previous_buffer_path = os.path.join(MODELS_DIR, f"{algorithm}_phase{phase - 1}_replay_buffer.pkl")
                    if os.path.exists(previous_buffer_path) and hasattr(model, "replay_buffer"):
                        try:
                            model.load_replay_buffer(previous_buffer_path)
                            print(f"已加载上一阶段的回放经验池: {previous_buffer_path}")
                        except Exception as e:
                            print(f"加载回放经验池失败: {e}")
        else:
            # 新阶段：检查是否需要从之前的阶段加载模型
            load_from_phase = phase_config.get("load_from_phase")
            if load_from_phase is not None:
                model_path = os.path.join(MODELS_DIR, f"{algorithm}_phase{load_from_phase}")
                if os.path.exists(f"{model_path}.zip"):
                    print(f"从阶段{load_from_phase}加载模型")
                    model = type(model).load(model_path, env=env)

                    # 更新模型的学习率为当前阶段配置的学习率
                    if hasattr(model, "learning_rate"):
                        model.learning_rate = model_params["learning_rate"]
                        print(f"已更新学习率为当前阶段设置: {model_params['learning_rate']}")

                    # 尝试加载对应的经验回放池
                    buffer_path = os.path.join(MODELS_DIR, f"{algorithm}_phase{load_from_phase}_replay_buffer.pkl")
                    if os.path.exists(buffer_path) and hasattr(model, "replay_buffer"):
                        try:
                            model.load_replay_buffer(buffer_path)
                            print(f"已加载阶段{load_from_phase}的回放经验池")
                        except Exception as e:
                            print(f"加载回放经验池失败: {e}")
                else:
                    print(f"警告: 未找到阶段{load_from_phase}的模型，使用新初始化的模型")

        # 初始化阶段训练参数
        base_steps = base_timesteps_per_phase
        # 应用难度因子调整基础步数
        difficulty_factor = phase_config.get("difficulty_factor", 1.0)
        if phase_config['reward_threshold'] is not None:
            total_timesteps = int(base_steps * difficulty_factor)
        else:
            total_timesteps = int(base_steps)

        # 恢复训练状态
        total_steps = 0
        if resume and phase == start_phase:
            if "total_steps_in_phase" in state:
                total_steps = state["total_steps_in_phase"]

        # 创建日志和模型保存路径
        log_path = os.path.join(LOGS_DIR, f"eval_phase{phase}.csv")
        best_model_path = os.path.join(MODELS_DIR, f"{algorithm}_phase{phase}_best")
        latest_model_path = os.path.join(MODELS_DIR, f"{algorithm}_phase{phase}_latest")

        # 创建评估回调
        eval_callback = CurriculumEvalCallback(
            eval_env=eval_env,
            phase=phase,
            reward_threshold=phase_config["reward_threshold"],
            std_threshold_ratio=0.2,  # 标准差不超过平均值的20%
            n_eval_episodes=10,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_path,
            deterministic=True,
            min_delta=1.0,
            verbose=1
        )

        # 创建保存最新模型的回调
        latest_model_callback = SaveLatestModelCallback(
            save_path=latest_model_path,
            save_freq=eval_freq
        )

        # 合并所有回调
        all_callbacks = [eval_callback, latest_model_callback]

        # 创建检查点回调
        checkpoint_dir = os.path.join(MODELS_DIR, f"checkpoints_phase{phase}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_callback = SaveCheckpointCallback(
            algorithm=algorithm,
            save_dir=checkpoint_dir,
            save_freq=eval_freq * 5,
            max_checkpoints=5
        )
        all_callbacks.append(checkpoint_callback)

        # 创建学习率调度回调
        lr_schedule = {
            0: model_params["learning_rate"],  # 从0步开始使用配置的学习率
            100000: model_params["learning_rate"] * 0.1,  # 从100000步开始使用学习率的0.1倍
            200000: model_params["learning_rate"] * (0.1 ** 2),  # 从200000步开始使用学习率的0.01倍
            300000: model_params["learning_rate"] * (0.1 ** 3)  # 从300000步开始使用学习率的0.001倍
        }
        learning_rate_callback = StepwiseLRSchedulerCallback(lr_schedule, verbose=1)
        all_callbacks.append(learning_rate_callback)

        # 计算剩余训练步数
        remaining_steps = total_timesteps
        if resume and phase == start_phase:
            if phase_set is not None and phase == phase_set + 1:
                # 如果是强制跳过阶段后的第一个阶段，使用完整的训练步数
                remaining_steps = total_timesteps
                total_steps = 0  # 重置总步数，因为是新阶段的开始
                print(f"强制跳过阶段{phase_set}后，使用完整训练步数: {remaining_steps}")
            else:
                # 正常恢复训练的情况
                remaining_steps = max(0, total_timesteps - total_steps)
                print(f"恢复训练，剩余步数: {remaining_steps}")

        print(f"训练步数: {remaining_steps} (总计划: {total_timesteps})")

        # 训练模型
        try:
            # 当是新训练或强制跳过阶段后的第一个阶段时，重置步数计数器
            reset_num_timesteps = (phase != start_phase or (phase_set is not None and phase == phase_set + 1))
            model.learn(
                total_timesteps=remaining_steps,
                callback=all_callbacks,
                tb_log_name=f"{algorithm}_phase{phase}",
                reset_num_timesteps=reset_num_timesteps
            )
        except EarlyStoppingException as e:
            print(f"早停: {e}")
            pass

        # 无论是正常完成训练还是早停，都保存最终模型
        latest_model_path = os.path.join(MODELS_DIR, f"{algorithm}_phase{phase}_latest")
        model.save(latest_model_path)
        if hasattr(env, "save"):
            env.save(f"{latest_model_path}_vecnorm.pkl")
        print(f"阶段{phase}训练结束，已保存最终模型到 {latest_model_path}")

        # 更新总步数
        total_steps += remaining_steps

        # 评估当前性能
        phase_complete[phase] = eval_callback.get_phase_complete()

        if phase_complete[phase]:
            print(
                f"阶段{phase}完成! 平均奖励: {eval_callback.last_mean_reward:.2f}, "
                f"阈值: {phase_config['reward_threshold']:.2f}")
        else:
            print(
                f"阶段{phase}未完成. 最佳平均奖励: {eval_callback.best_mean_reward:.2f}")
            # 如果当前阶段未完成，停止训练
            print(f"阶段{phase}未达到目标奖励阈值，停止训练")

            break

        # 保存训练状态
        save_training_state(
            algorithm=algorithm,
            phase_complete=phase_complete,
            phase=phase,
            timesteps_per_phase=base_timesteps_per_phase,
            model=model,
            env=env,
            total_steps=total_steps
        )

        # 保存最终模型
        final_model_path = os.path.join(MODELS_DIR, f"{algorithm}_phase{phase}")
        model.save(final_model_path)
        print(f"阶段{phase}最终模型已保存: {final_model_path}")

        # 保存经验回放池
        if hasattr(model, "replay_buffer"):
            replay_buffer_path = os.path.join(MODELS_DIR, f"{algorithm}_phase{phase}_replay_buffer.pkl")
            try:
                model.save_replay_buffer(replay_buffer_path)
                print(f"阶段{phase}经验回放池已保存: {replay_buffer_path}")
            except Exception as e:
                print(f"保存经验回放池失败: {e}")

        # 保存环境状态
        if hasattr(env, "save"):
            env_path = os.path.join(MODELS_DIR, f"vec_normalize_{algorithm}_phase{phase}.pkl")
            try:
                env.save(env_path)
                print(f"环境状态已保存: {env_path}")
            except Exception as e:
                print(f"保存环境状态失败: {e}")

    print("课程学习完成！")

    # 返回最终的评估结果
    final_results = {f"phase{i}_complete": phase_complete[i] for i in range(1, total_phases + 1)}
    return final_results
