# -*- coding: utf-8 -*-
"""
@File    : curriculum.py
@Author  : zhangjian
@Desc    : 课程学习实现
"""

import pickle
import datetime
from copy import deepcopy
import glob

import random
import torch
from datetime import datetime

from stable_baselines3 import PPO, SAC, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config.maze_search.default_config import MODELS_DIR, LOGS_DIR
from training.callbacks import *

# 创建保存最佳模型的目录
best_model_dir = os.path.join(MODELS_DIR, f"best_models")
os.makedirs(best_model_dir, exist_ok=True)

# 定义更复杂的神经网络配置
policy_kwargs = {
    "net_arch": {
        "pi": [256, 256, 256, 256],  # 策略网络
        "qf": [256, 256, 256, 256]  # Q函数网络
    }
}


def create_env_from_config(env_class,
                           env_config,
                           is_eval=False,
                           verbose=False,
                           seed=None):
    """
    根据配置创建环境

    Args:
        env_class: 环境类
        env_config: 环境配置字典
        is_eval: 是否为评估环境
        verbose:
    Returns:
        创建好的环境

    """
    # 复制配置以避免修改原始配置
    env_config = deepcopy(env_config)

    # 创建基础环境
    # 创建环境时传递完整的env_config
    env = env_class(
        maze_size=env_config.get("maze_size", (7, 7)),
        verbose=verbose,  # 可以根据需要调整
        env_config=env_config,  # 传递完整配置
        is_eval=is_eval,
        seed=seed
    )

    # 设置课程阶段
    env.curriculum_phase = env_config.get("curriculum_phase", 1)

    # 包装环境
    env = Monitor(env)  # 返回真实奖励
    env = DummyVecEnv([lambda: env])

    # 关键修改：确保训练和评估使用相同的归一化统计数据
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.,
        clip_reward=10.,
        gamma=0.99,
        training=not is_eval  # 评估时不更新统计数据
    )

    return env


def create_model_from_config(algorithm, env, model_params, seed):
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
        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=LOGS_DIR, seed=seed, **model_params,
                    policy_kwargs=policy_kwargs)
    elif algorithm == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOGS_DIR, seed=seed, **model_params)
    else:
        raise ValueError(f"不支持的算法: {algorithm}")

    return model


def save_training_state(algorithm, phase_complete, phase, timesteps_per_phase, model=None, env=None, total_steps=0,
                        attempt=0, phase_config=None, callbacks=None):
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
        callbacks: 训练回调函数列表 (可选)
    """
    state_dir = os.path.join(MODELS_DIR, "states")
    os.makedirs(state_dir, exist_ok=True)

    # 基本训练状态
    state = {
        "phase_complete": phase_complete,
        "current_phase": phase,
        "timesteps_per_phase": timesteps_per_phase,
        "total_steps_in_phase": total_steps,
        "attempt": attempt,
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
            print(f"当前学习率已保存: {state['learning_rate']}")
        else:
            print("警告：模型没有学习率属性，学习率未保存")

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

    # 保存回调状态
    if callbacks is not None:
        state["callbacks"] = {}
        for callback in callbacks:
            if hasattr(callback, "__class__") and callback.__class__.__name__ == "LinearLRDecayCallback":
                state["callbacks"]["LinearLRDecayCallback"] = {
                    "decay_start_step": callback.decay_start_step,
                    "decay_end_step": callback.decay_end_step,
                    "final_lr_fraction": callback.final_lr_fraction,
                    "decay_by_time_steps": callback.decay_by_time_steps,
                    "initial_lr": callback.initial_lr,
                    "current_lr": getattr(callback, "current_lr", callback.initial_lr),
                    "n_calls": callback.n_calls
                }
                print(f"学习率衰减回调状态已保存: 当前步数 {callback.n_calls}")

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
            "max_attempts": phase_config.get("max_attempts", 3),
            "step_increase_factor": phase_config.get("step_increase_factor", 1.5)
        }
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

            # 加载并显示学习率衰减回调状态
            if "callbacks" in state and "LinearLRDecayCallback" in state["callbacks"]:
                lr_decay_state = state["callbacks"]["LinearLRDecayCallback"]
                print(f"  - 学习率衰减回调状态:")
                print(f"    - 当前步数: {lr_decay_state.get('n_calls', 0)}")
                print(f"    - 当前学习率: {lr_decay_state.get('current_lr', '未知')}")
                print(f"    - 衰减开始步数: {lr_decay_state.get('decay_start_step', '未知')}")
                print(f"    - 衰减结束步数: {lr_decay_state.get('decay_end_step', '未知')}")
                print(f"    - 最终学习率比例: {lr_decay_state.get('final_lr_fraction', '未知')}")

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


def save_training_state_on_interrupt(
        algorithm, model, env, phase, total_steps, model_dir, verbose=1, callbacks=None
):
    """
    在接收到中断信号时保存训练状态
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    interrupt_model_path = os.path.join(model_dir, f"{algorithm}_phase{phase}_interrupt_{timestamp}")
    model.save(interrupt_model_path)
    if verbose >= 1:
        print(f"中断训练，模型已保存到: {interrupt_model_path}")

    # 保存经验回放池
    if hasattr(model, "replay_buffer"):
        replay_buffer_path = os.path.join(model_dir,
                                          f"{algorithm}_phase{phase}_replay_buffer_interrupt_{timestamp}.pkl")
        try:
            model.save_replay_buffer(replay_buffer_path)
            if verbose >= 1:
                print(f"经验回放池已保存到: {replay_buffer_path}")
        except Exception as e:
            print(f"保存经验回放池失败: {e}")

    # 保存环境状态
    if hasattr(env, "save"):
        env_path = os.path.join(model_dir, f"vec_normalize_{algorithm}_phase{phase}_interrupt_{timestamp}.pkl")
        try:
            env.save(env_path)
            if verbose >= 1:
                print(f"环境状态已保存到: {env_path}")
        except Exception as e:
            print(f"保存环境状态失败: {e}")

    # 保存训练状态（步数、随机数状态等）
    state = {
        "total_steps": total_steps,
        "random_state": {
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
    }
    # 保存当前学习率
    if hasattr(model, "learning_rate"):
        state["learning_rate"] = model.learning_rate
        if verbose >= 1:
            print(f"当前学习率已保存: {state['learning_rate']}")

    # 保存学习率衰减回调状态
    if callbacks:
        for callback in callbacks:
            if hasattr(callback, "__class__") and callback.__class__.__name__ == "LinearLRDecayCallback":
                state["lr_decay_callback"] = {
                    "n_calls": callback.n_calls if hasattr(callback, "n_calls") else 0,
                    "decay_start_step": callback.decay_start_step if hasattr(callback, "decay_start_step") else 0,
                    "decay_end_step": callback.decay_end_step if hasattr(callback, "decay_end_step") else 200000,
                    "initial_lr": callback.initial_lr if hasattr(callback, "initial_lr") else model.learning_rate,
                    "final_lr_fraction": callback.final_lr_fraction if hasattr(callback, "final_lr_fraction") else 0.1
                }
                if verbose >= 1:
                    print(f"学习率衰减回调状态已保存: 当前步数 {state['lr_decay_callback']['n_calls']}")

    try:
        state["random_state"]["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["random_state"]["torch_cuda"] = torch.cuda.get_rng_state_all()
    except Exception as e:
        print(f"保存PyTorch随机数状态失败: {e}")

    state_path = os.path.join(model_dir, f"training_state_phase{phase}_interrupt_{timestamp}.pkl")
    import pickle
    with open(state_path, "wb") as f:
        pickle.dump(state, f)
    if verbose >= 1:
        print(f"训练状态已保存到: {state_path}")


def load_training_state_for_resume(model_dir, phase, verbose=1):
    """
    加载上次中断的训练状态
    """
    import glob
    import pickle
    state_files = glob.glob(os.path.join(model_dir, f"training_state_phase{phase}_interrupt_*.pkl"))
    if not state_files:
        if verbose >= 1:
            print("未找到中断状态文件，从头开始训练或加载最新模型")
        return None

    # 选择最新的状态文件
    latest_state_file = max(state_files, key=os.path.getctime)
    with open(latest_state_file, "rb") as f:
        state = pickle.load(f)
    if verbose >= 1:
        print(f"加载中断训练状态: {latest_state_file}")
        if "learning_rate" in state:
            print(f"加载学习率: {state['learning_rate']}")
    return state


def load_model_and_state(
        model,
        env,
        algorithm,
        phase,
        model_dir,
        verbose=0,
        resume=False,
        state_loader_func=None,
        default_steps=0
):
    """
    加载模型、经验回放池、环境状态和训练状态的通用函数。

    Args:
        model: 模型对象
        env: 环境对象
        algorithm: 算法名称（如 "SAC"）
        phase: 当前阶段编号
        model_dir: 模型保存目录
        verbose: 日志详细程度
        resume: 是否恢复训练
        state_loader_func: 自定义状态加载函数，默认为 None
        default_steps: 默认已完成步数
    Returns:
        tuple: (loaded_steps, updated_model, updated_env)
    """
    loaded_steps = default_steps
    if not resume:
        if verbose >= 1:
            print("不进行恢复训练，从头开始")
        return loaded_steps, model, env

    # 尝试加载训练状态
    state = None
    if state_loader_func:
        state = state_loader_func(model_dir, phase, verbose)
    if state:
        loaded_steps = state.get("total_steps", 0)
        if verbose >= 1:
            print(f"从状态文件加载步数: {loaded_steps}")
        # 恢复随机数状态
        try:
            np.random.set_state(state["random_state"]["numpy"])
            random.setstate(state["random_state"]["python"])
            if "torch" in state["random_state"]:
                torch.set_rng_state(state["random_state"]["torch"])
            if "torch_cuda" in state["random_state"] and torch.cuda.is_available():
                if isinstance(state["random_state"]["torch_cuda"], list):
                    torch.cuda.set_rng_state_all(state["random_state"]["torch_cuda"])
                else:
                    torch.cuda.set_rng_state(state["random_state"]["torch_cuda"])
            if verbose >= 1:
                print("已恢复随机数状态")
        except Exception as e:
            print(f"恢复随机数状态失败: {e}")

    # 尝试加载中断模型
    model_files = glob.glob(os.path.join(model_dir, f"{algorithm}_phase{phase}_interrupt_*.zip"))
    if model_files:
        latest_model_path = max(model_files, key=os.path.getctime)
        try:
            model.load(latest_model_path)
            model_steps = 0
            if hasattr(model, "num_timesteps"):
                model_steps = model.num_timesteps
                if verbose >= 1:
                    print(f"从中断模型 {latest_model_path} 继续训练，模型步数: {model_steps}")
                if model_steps == 0 and loaded_steps > 0:
                    if verbose >= 1:
                        print(f"模型步数为0，使用状态文件步数: {loaded_steps}")
                    try:
                        model.num_timesteps = loaded_steps
                        if verbose >= 1:
                            print(f"已手动设置模型步数为: {loaded_steps}")
                    except Exception as e:
                        if verbose >= 1:
                            print(f"无法手动设置模型步数: {e}")
                else:
                    loaded_steps = max(loaded_steps, model_steps)
            else:
                if verbose >= 1:
                    print(f"从中断模型 {latest_model_path} 继续训练")
            # 恢复学习率
            if state and "learning_rate" in state:
                try:
                    model.learning_rate = state["learning_rate"]
                    if verbose >= 1:
                        print(f"已恢复学习率: {state['learning_rate']}")
                except Exception as e:
                    print(f"无法恢复学习率: {e}")
        except Exception as e:
            print(f"加载中断模型失败: {e}")
    else:
        if verbose >= 1:
            print(f"未找到中断模型文件，尝试加载最新模型")
        latest_model_path = os.path.join(model_dir, f"{algorithm}_phase{phase}_latest")
        if os.path.exists(latest_model_path + ".zip"):
            try:
                model.load(latest_model_path)
                model_steps = 0
                if hasattr(model, "num_timesteps"):
                    model_steps = model.num_timesteps
                    if verbose >= 1:
                        print(f"从 {latest_model_path} 加载模型以继续训练，模型步数: {model_steps}")
                    if model_steps == 0 and loaded_steps > 0:
                        if verbose >= 1:
                            print(f"模型步数为0，使用状态文件步数: {loaded_steps}")
                        try:
                            model.num_timesteps = loaded_steps
                            if verbose >= 1:
                                print(f"已手动设置模型步数为: {loaded_steps}")
                        except Exception as e:
                            if verbose >= 1:
                                print(f"无法手动设置模型步数: {e}")
                    else:
                        loaded_steps = max(loaded_steps, model_steps)
                else:
                    if verbose >= 1:
                        print(f"从 {latest_model_path} 加载模型以继续训练")
                # 恢复学习率
                if state and "learning_rate" in state:
                    try:
                        model.learning_rate = state["learning_rate"]
                        if verbose >= 1:
                            print(f"已恢复学习率: {state['learning_rate']}")
                    except Exception as e:
                        print(f"无法恢复学习率: {e}")
            except Exception as e:
                print(f"加载最新模型失败: {e}")
        else:
            if verbose >= 1:
                print(f"未找到模型 {latest_model_path}，从头开始训练")

    # 加载经验回放池
    replay_buffer_files = glob.glob(
        os.path.join(model_dir, f"{algorithm}_phase{phase}_replay_buffer_interrupt_*.pkl"))
    if replay_buffer_files and hasattr(model, "load_replay_buffer"):
        latest_replay_buffer_path = max(replay_buffer_files, key=os.path.getctime)
        try:
            model.load_replay_buffer(latest_replay_buffer_path)
            if verbose >= 1:
                print(f"从 {latest_replay_buffer_path} 加载经验回放池")
        except Exception as e:
            print(f"加载经验回放池失败: {e}")
    else:
        if verbose >= 1:
            print("未找到经验回放池文件或模型不支持加载回放池，训练将从空回放池开始")

    # 加载环境状态
    env_files = glob.glob(os.path.join(model_dir, f"vec_normalize_{algorithm}_phase{phase}_interrupt_*.pkl"))
    if env_files and hasattr(env, "load"):
        latest_env_path = max(env_files, key=os.path.getctime)
        try:
            from stable_baselines3.common.vec_env import VecNormalize
            if isinstance(env, VecNormalize):
                base_env = env.venv if hasattr(env, 'venv') else env
                env = VecNormalize.load(latest_env_path, base_env)
                if verbose >= 1:
                    print(f"从 {latest_env_path} 加载环境归一化状态")
            else:
                try:
                    env.load(latest_env_path)
                    if verbose >= 1:
                        print(f"从 {latest_env_path} 加载环境状态")
                except TypeError as te:
                    if "missing 1 required positional argument: 'venv'" in str(te):
                        env.load(latest_env_path, env)
                        if verbose >= 1:
                            print(f"从 {latest_env_path} 加载环境状态（提供 venv 参数）")
                    else:
                        raise te
        except Exception as e:
            print(f"加载环境状态失败: {e}")
    else:
        if verbose >= 1:
            print("未找到环境状态文件或环境不支持加载状态，跳过加载")

    return loaded_steps, model, env


def save_model_and_state(
        model,
        env,
        algorithm,
        phase,
        model_dir,
        suffix="latest",
        verbose=0
):
    """
    保存模型、经验回放池和环境状态的通用函数。

    Args:
        model: 模型对象
        env: 环境对象
        algorithm: 算法名称（如 "SAC"）
        phase: 当前阶段编号
        model_dir: 模型保存目录
        suffix: 文件后缀（如 "latest" 或 "interrupt_20250627"）
        verbose: 日志详细程度
    """
    model_path = os.path.join(model_dir, f"{algorithm}_phase{phase}_{suffix}")
    try:
        model.save(model_path)
        if verbose >= 1:
            print(f"模型已保存到: {model_path}")
    except Exception as e:
        print(f"保存模型失败: {e}")

    if hasattr(env, "save"):
        env_path = os.path.join(model_dir, f"vec_normalize_{algorithm}_phase{phase}_{suffix}.pkl")
        try:
            env.save(env_path)
            if verbose >= 1:
                print(f"环境状态已保存到: {env_path}")
        except Exception as e:
            print(f"保存环境状态失败: {e}")

    if hasattr(model, "replay_buffer") and hasattr(model, "save_replay_buffer"):
        replay_buffer_path = os.path.join(model_dir, f"{algorithm}_phase{phase}_replay_buffer_{suffix}.pkl")
        try:
            model.save_replay_buffer(replay_buffer_path)
            if verbose >= 1:
                print(f"经验回放池已保存到: {replay_buffer_path}")
        except Exception as e:
            print(f"保存经验回放池失败: {e}")


def save_training_state_on_interrupt_wrapper(
        model,
        env,
        algorithm,
        phase,
        model_dir,
        verbose=0,
        state_saver_func=None,
        callbacks=None
):
    """
    处理中断时保存训练状态的通用函数。

    Args:
        model: 模型对象
        env: 环境对象
        algorithm: 算法名称
        phase: 当前阶段编号
        model_dir: 模型保存目录
        verbose: 日志详细程度
        state_saver_func: 自定义状态保存函数，默认为 None
        callbacks: 回调函数列表，用于保存回调状态
    Returns:
        int: 当前步数
    """
    current_steps = 0
    if hasattr(model, "num_timesteps"):
        current_steps = model.num_timesteps
    if verbose >= 1:
        print(f"保存中断状态，当前模型步数: {current_steps}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"interrupt_{timestamp}"
    save_model_and_state(model, env, algorithm, phase, model_dir, suffix=suffix, verbose=verbose)

    # 保存回调状态（如果没有自定义的 state_saver_func）
    if state_saver_func:
        state_saver_func(
            algorithm=algorithm,
            model=model,
            env=env,
            phase=phase,
            total_steps=current_steps,
            model_dir=model_dir,
            verbose=verbose,
            callbacks=callbacks
        )
    else:
        # 如果没有自定义保存函数，直接在这里保存回调状态
        if verbose >= 1:
            print("未提供自定义状态保存函数，使用默认逻辑保存回调状态")
        state = {
            "algorithm": algorithm,
            "current_phase": phase,
            "total_steps": current_steps,
            "save_time": {"datetime": timestamp},
            "callbacks": {}
        }
        if callbacks:
            for callback in callbacks:
                if hasattr(callback, "__class__") and callback.__class__.__name__ == "LinearLRDecayCallback":
                    print(f"LinearLRDecayCallback可以调用")
                    state["callbacks"]["LinearLRDecayCallback"] = {
                        "n_calls": callback.n_calls,
                        "current_lr": getattr(callback, "current_lr", None),
                        "decay_start_step": getattr(callback, "decay_start_step", None),
                        "decay_end_step": getattr(callback, "decay_end_step", None),
                        "final_lr_fraction": getattr(callback, "final_lr_fraction", None),
                        "initial_lr": getattr(callback, "initial_lr", None)
                    }
                    print(state)
                    if verbose >= 1:
                        print(f"学习率衰减回调状态已保存: 当前步数 {callback.n_calls}")

        # 保存状态到文件
        state_path = os.path.join(model_dir, "states", f"{algorithm}_training_state_interrupt_{timestamp}.pkl")
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, "wb") as f:
            pickle.dump(state, f)
        if verbose >= 1:
            print(f"训练状态已保存到: {state_path}")

    return current_steps
