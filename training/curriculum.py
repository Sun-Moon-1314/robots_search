# -*- coding: utf-8 -*-
"""
@File    : curriculum.py
@Author  : zhangjian
@Desc    : 课程学习实现
"""

import random
from graphene.types.scalars import MAX_INT

from config.maze_search.default_config import MODELS_DIR, LOGS_DIR
from training.common import load_training_state, create_env_from_config, create_model_from_config, save_training_state
from training.callbacks import *

# 创建保存最佳模型的目录
best_model_dir = os.path.join(MODELS_DIR, f"best_models")
os.makedirs(best_model_dir, exist_ok=True)


def resume_processor(resume, phase, start_phase, total_phases, model, model_params, env, algorithm, phase_set,
                     phase_config):
    """
    断点恢复处理
    :param resume:
    :param phase:
    :param start_phase:
    :param total_phases:
    :param model:
    :param model_params:
    :param env:
    :param algorithm:
    :param phase_set:
    :param phase_config:
    :return:
    """
    state = load_training_state(algorithm, total_phases)
    if resume and phase == start_phase:
        # 恢复训练：优先加载保存的训练状态中的模型
        if "model_path" in state and os.path.exists(f"{state['model_path']}.zip"):
            print(f"从保存的训练状态加载模型: {state['model_path']}")
            try:
                # 使用load方法加载模型，但保留当前阶段的学习率设置
                model = type(model).load(state["model_path"], env=env)

                # 更新模型的学习率为当前阶段配置的学习率
                new_lr = model_params["learning_rate"]

                # 更新模型的学习率为当前阶段配置的学习率
                if hasattr(model, "learning_rate"):
                    model.learning_rate = new_lr

                # 2. 更新策略网络(Actor)优化器的学习率
                if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
                    for param_group in model.policy.optimizer.param_groups:
                        param_group['lr'] = new_lr

                # 3. 如果使用SAC，还需要更新critic优化器的学习率
                if hasattr(model, "critic") and hasattr(model.critic, "optimizer"):
                    for param_group in model.critic.optimizer.param_groups:
                        param_group['lr'] = new_lr

                # 4. 如果有第二个critic网络
                if hasattr(model, "critic_target") and hasattr(model.critic_target, "optimizer"):
                    for param_group in model.critic_target.optimizer.param_groups:
                        param_group['lr'] = new_lr

                # 5. 如果有熵系数优化器
                if hasattr(model, "ent_coef_optimizer"):
                    for param_group in model.ent_coef_optimizer.param_groups:
                        param_group['lr'] = new_lr

                print("模型加载成功")

                # 尝试恢复环境状态（如果有），但不强制要求
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
                        print(f"加载环境状态失败，但继续训练: {e}")

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

        # 如果没有找到训练状态中的模型或加载失败，则按新的优先级尝试加载
        if not ("model_path" in state and os.path.exists(f"{state['model_path']}.zip")):
            # 修改加载优先级：先尝试加载phase模型，再尝试加载best模型，不加载latest模型
            phase_path = os.path.join(MODELS_DIR, f"{algorithm}_phase{phase}")
            best_path = os.path.join(MODELS_DIR, f"best_model")

            # 处理强制跳过阶段的情况
            if phase_set is not None and phase == start_phase:
                # 尝试加载前一阶段的模型作为起点
                prev_phase_path = os.path.join(MODELS_DIR, f"{algorithm}_phase{phase_set}")
                prev_best_path = os.path.join(MODELS_DIR, f"best_model")

                # 按新的优先级尝试加载前一阶段的模型
                model_loaded = False
                for path_name, path in [
                    ("完成模型", prev_phase_path),
                    ("最佳模型", prev_best_path)
                ]:
                    if os.path.exists(f"{path}.zip"):
                        print(f"强制跳过：从阶段{phase_set}的{path_name}加载")
                        print(f"加载目录为:{path}.zip")
                        model = type(model).load(path, env=env)

                        # 更新模型的学习率为当前阶段配置的学习率
                        new_lr = model_params["learning_rate"]

                        # 更新模型的学习率为当前阶段配置的学习率
                        if hasattr(model, "learning_rate"):
                            model.learning_rate = new_lr

                        # 2. 更新策略网络(Actor)优化器的学习率
                        if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
                            for param_group in model.policy.optimizer.param_groups:
                                param_group['lr'] = new_lr

                        # 3. 如果使用SAC，还需要更新critic优化器的学习率
                        if hasattr(model, "critic") and hasattr(model.critic, "optimizer"):
                            for param_group in model.critic.optimizer.param_groups:
                                param_group['lr'] = new_lr

                        # 4. 如果有第二个critic网络
                        if hasattr(model, "critic_target") and hasattr(model.critic_target, "optimizer"):
                            for param_group in model.critic_target.optimizer.param_groups:
                                param_group['lr'] = new_lr

                        # 5. 如果有熵系数优化器
                        if hasattr(model, "ent_coef_optimizer"):
                            for param_group in model.ent_coef_optimizer.param_groups:
                                param_group['lr'] = new_lr

                        # 尝试加载对应的环境归一化状态，但不强制要求
                        vec_norm_path = f"{path}_vecnorm.pkl"
                        if os.path.exists(vec_norm_path):
                            try:
                                from stable_baselines3.common.vec_env import VecNormalize
                                env = VecNormalize.load(vec_norm_path, env)
                                print(f"已加载环境归一化状态: {vec_norm_path}")
                            except Exception as e:
                                print(f"加载环境归一化状态失败，但继续训练: {e}")
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
                        new_lr = model_params["learning_rate"]

                        # 更新模型的学习率为当前阶段配置的学习率
                        if hasattr(model, "learning_rate"):
                            model.learning_rate = new_lr

                        # 2. 更新策略网络(Actor)优化器的学习率
                        if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
                            for param_group in model.policy.optimizer.param_groups:
                                param_group['lr'] = new_lr

                        # 3. 如果使用SAC，还需要更新critic优化器的学习率
                        if hasattr(model, "critic") and hasattr(model.critic, "optimizer"):
                            for param_group in model.critic.optimizer.param_groups:
                                param_group['lr'] = new_lr

                        # 4. 如果有第二个critic网络
                        if hasattr(model, "critic_target") and hasattr(model.critic_target, "optimizer"):
                            for param_group in model.critic_target.optimizer.param_groups:
                                param_group['lr'] = new_lr

                        # 5. 如果有熵系数优化器
                        if hasattr(model, "ent_coef_optimizer"):
                            for param_group in model.ent_coef_optimizer.param_groups:
                                param_group['lr'] = new_lr

                        # 尝试加载对应的环境归一化状态，但不强制要求
                        vec_norm_path = f"{checkpoint_path}_vecnorm.pkl"
                        if os.path.exists(vec_norm_path):
                            try:
                                from stable_baselines3.common.vec_env import VecNormalize
                                env = VecNormalize.load(vec_norm_path, env)
                                print(f"已加载环境归一化状态: {vec_norm_path}")
                            except Exception as e:
                                print(f"加载环境归一化状态失败，但继续训练: {e}")
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
                # 正常情况下按新的优先级尝试加载当前阶段的模型
                model_loaded = False
                for path_name, path in [
                    ("完成模型", phase_path),
                    ("最佳模型", best_path)
                ]:
                    if os.path.exists(f"{path}.zip"):
                        print(f"恢复训练：加载阶段{phase}的{path_name}")
                        model = type(model).load(path, env=env)

                        # 更新模型的学习率为当前阶段配置的学习率
                        new_lr = model_params["learning_rate"]

                        # 更新模型的学习率为当前阶段配置的学习率
                        if hasattr(model, "learning_rate"):
                            model.learning_rate = new_lr

                        # 2. 更新策略网络(Actor)优化器的学习率
                        if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
                            for param_group in model.policy.optimizer.param_groups:
                                param_group['lr'] = new_lr

                        # 3. 如果使用SAC，还需要更新critic优化器的学习率
                        if hasattr(model, "critic") and hasattr(model.critic, "optimizer"):
                            for param_group in model.critic.optimizer.param_groups:
                                param_group['lr'] = new_lr

                        # 4. 如果有第二个critic网络
                        if hasattr(model, "critic_target") and hasattr(model.critic_target, "optimizer"):
                            for param_group in model.critic_target.optimizer.param_groups:
                                param_group['lr'] = new_lr

                        # 5. 如果有熵系数优化器
                        if hasattr(model, "ent_coef_optimizer"):
                            for param_group in model.ent_coef_optimizer.param_groups:
                                param_group['lr'] = new_lr

                        print(f"已更新所有优化器的学习率为当前阶段设置: {new_lr}")

                        # 6. 验证学习率是否真的更新了
                        if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
                            actual_lr = model.policy.optimizer.param_groups[0]['lr']
                            print(f"Actor优化器的实际学习率: {actual_lr}")

                        # 尝试加载对应的环境归一化状态，但不强制要求
                        vec_norm_path = f"{path}_vecnorm.pkl"
                        if os.path.exists(vec_norm_path):
                            try:
                                from stable_baselines3.common.vec_env import VecNormalize
                                env = VecNormalize.load(vec_norm_path, env)
                                print(f"已加载环境归一化状态: {vec_norm_path}")
                            except Exception as e:
                                print(f"加载环境归一化状态失败，但继续训练: {e}")
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

                        # 尝试加载对应的环境归一化状态，但不强制要求
                        vec_norm_path = f"{checkpoint_path}_vecnorm.pkl"
                        if os.path.exists(vec_norm_path):
                            try:
                                from stable_baselines3.common.vec_env import VecNormalize
                                env = VecNormalize.load(vec_norm_path, env)
                                print(f"已加载环境归一化状态: {vec_norm_path}")
                            except Exception as e:
                                print(f"加载环境归一化状态失败，但继续训练: {e}")
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
            best_path = os.path.join(MODELS_DIR, f"best_model")

            # 按新的优先级尝试加载模型
            model_loaded = False
            for path_name, path in [
                ("完成模型", model_path),
                ("最佳模型", best_path)
            ]:
                if os.path.exists(f"{path}.zip"):
                    print(f"从{path_name}加载")
                    model = type(model).load(path, env=env)

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

                    model_loaded = True
                    break

            if not model_loaded:
                print(f"警告: 未找到阶段{load_from_phase}的模型或最佳模型，使用新初始化的模型")


def train_with_curriculum(env_class, curriculum_config, resume=False, phase_set=None, verbose=False):
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
    start_phase = 1
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
        seed = phase_config["seed"]
        performance_thresholds = phase_config["performance_thresholds"]
        step_thresholds = performance_thresholds["step_thresholds"]
        print(f"阶段{phase}: {phase_config['name']}...")

        # 创建训练环境
        env = create_env_from_config(env_class, phase_config["env_config"],
                                     is_eval=False, verbose=verbose, seed=seed)

        # 创建评估环境
        eval_env = create_env_from_config(env_class, phase_config["env_config"], is_eval=True, verbose=verbose,
                                          seed=seed)

        # 获取当前阶段的模型参数
        model_params = phase_config["model_params"][algorithm]
        if verbose >= 2:
            print(f"当前阶段{phase}, 当前学习率为:{model_params['learning_rate']}")

        # 创建模型
        model = create_model_from_config(algorithm, env, model_params, seed)
        if verbose >= 2:
            print(f"当前阶段{phase}, 模型创建后的学习率为:{model.learning_rate}")
        # 处理模型加载逻辑
        resume_processor(resume, phase, start_phase, total_phases, model, model_params, env, algorithm, phase_set,
                         phase_config)

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
        save_log_callback = SaveModelLogCallback(verbose=1)
        # 创建评估回调
        eval_callback = CurriculumEvalCallback(
            eval_env=eval_env,
            phase=phase,
            performance_thresholds=phase_config.get("performance_thresholds"),  # 添加性能阈值参数
            callback_on_new_best=save_log_callback,
            std_threshold_ratio=0.2,  # 标准差不超过平均值的20%
            n_eval_episodes=10,
            early_stop_patience=MAX_INT,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_path,
            deterministic=True,
            min_delta=1.0,
            verbose=1,
            check_direction=False,  # 启用方向检查
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
            max_checkpoints=5
        )
        all_callbacks.append(checkpoint_callback)
        # 创建学习率调度回调
        learning_rate_callback = LinearLRDecayCallback(
            verbose=2,
            decay_start_step=0,  # 从第0步开始衰减
            decay_end_step=200000,  # 到第90000步结束衰减
            final_lr_fraction=0.1  # 最终学习率为初始学习率的10%
        )
        # 创建球位置分布回调
        all_callbacks.append(learning_rate_callback)
        # 小球位置变化回调函数
        # ball_position_callback = BallPositionCallback(eval_env)
        # all_callbacks.append(ball_position_callback)
        # 创建回调函数
        reward_callback = RewardMetricsCallback(
            verbose=1,  # 日志详细程度
            log_freq=50,  # 每1000步记录一次数据
        )
        all_callbacks.append(reward_callback)
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
            pass

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
