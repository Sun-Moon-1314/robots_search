import signal
import os

from graphene.types.scalars import MAX_INT

from config.maze_search.curriculum_config import TRAINER_CONFIG
from config.maze_search.default_config import LOGS_DIR, MODELS_DIR
from training.callbacks import SaveModelLogCallback, CurriculumEvalCallback, SaveLatestModelCallback, \
    SaveCheckpointCallback, LinearLRDecayCallback, RewardMetricsCallback, EarlyStoppingException
from training.common import create_env_from_config, create_model_from_config, load_training_state_for_resume, \
    save_training_state_on_interrupt, save_training_state_on_interrupt_wrapper, save_model_and_state, \
    load_model_and_state


def train_single_phase(
        env_class,
        config=None,
        algorithm="SAC",
        phase=1,
        total_timesteps=None,
        log_dir=LOGS_DIR,
        model_dir=MODELS_DIR,
        verbose=0,
        resume=False,
        total_steps_in_phase=0,
        max_steps_threshold=70
):
    """
    单个阶段或独立训练智能体的函数，不包含课程学习逻辑
    """
    if config is None:
        config = TRAINER_CONFIG

    seed = config.get("seed", 100)
    eval_freq = config.get("eval_freq", 5000)
    performance_thresholds = config.get("performance_thresholds", {})

    # 创建训练和评估环境
    env = create_env_from_config(env_class, config["env_config"], is_eval=False, verbose=verbose, seed=seed)
    eval_env = create_env_from_config(env_class, config["env_config"], is_eval=True, verbose=verbose, seed=seed)

    # 创建模型
    model_params = config["model_params"][algorithm]
    if verbose >= 2:
        print(f"当前阶段{phase}, 当前学习率为:{model_params['learning_rate']}")
    model = create_model_from_config(algorithm, env, model_params, seed)
    if verbose >= 2:
        print(f"当前阶段{phase}, 模型创建后的学习率为:{model.learning_rate}")

    # 加载模型和状态
    loaded_steps, model, env = load_model_and_state(
        model=model,
        env=env,
        algorithm=algorithm,
        phase=phase,
        model_dir=model_dir,
        verbose=verbose,
        resume=resume,
        state_loader_func=load_training_state_for_resume,
        default_steps=total_steps_in_phase
    )

    # 设置总训练步数
    if total_timesteps is None:
        base_steps = config.get("timesteps_per_phase", 500000)
        total_timesteps = base_steps

    # 计算剩余步数
    remaining_steps = max(0, total_timesteps - loaded_steps) if resume else total_timesteps
    if verbose >= 1:
        print(f"训练步数: {remaining_steps} (总计划: {total_timesteps}, 已完成: {loaded_steps})")

    # 创建日志和模型保存路径
    log_path = os.path.join(log_dir, f"eval_phase{phase}.csv")
    best_model_path = os.path.join(model_dir, f"{algorithm}_phase{phase}_best")
    latest_model_path = os.path.join(model_dir, f"{algorithm}_phase{phase}_latest")

    # 创建回调
    save_log_callback = SaveModelLogCallback(verbose=1)
    eval_callback = CurriculumEvalCallback(
        eval_env=eval_env,
        phase=phase,
        performance_thresholds=performance_thresholds,
        callback_on_new_best=save_log_callback,
        std_threshold_ratio=0.2,
        n_eval_episodes=20,
        early_stop_patience=MAX_INT,
        eval_freq=eval_freq,
        log_path=log_path,
        best_model_save_path=best_model_path,
        deterministic=True,
        min_delta=1.0,
        verbose=1,
        check_direction=True,
        max_steps_threshold=max_steps_threshold
    )
    latest_model_callback = SaveLatestModelCallback(save_path=latest_model_path, save_freq=eval_freq)
    checkpoint_dir = os.path.join(model_dir, f"checkpoints_phase{phase}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = SaveCheckpointCallback(algorithm=algorithm, save_dir=checkpoint_dir, max_checkpoints=5)
    learning_rate_callback = LinearLRDecayCallback(
        verbose=1, decay_start_step=0, decay_end_step=200000, final_lr_fraction=0.1
    )
    reward_callback = RewardMetricsCallback(verbose=1, log_freq=1000)

    all_callbacks = [
        eval_callback,
        latest_model_callback,
        checkpoint_callback,
        learning_rate_callback,
        reward_callback
    ]

    # 恢复学习率衰减回调状态
    state = load_training_state_for_resume(model_dir, phase, verbose)
    if state and "lr_decay_callback" in state:
        lr_state = state["lr_decay_callback"]
        if hasattr(learning_rate_callback, "n_calls"):
            learning_rate_callback.n_calls = lr_state["n_calls"]
        if hasattr(learning_rate_callback, "initial_lr"):
            learning_rate_callback.initial_lr = lr_state["initial_lr"]
        if hasattr(learning_rate_callback, "decay_start_step"):
            learning_rate_callback.decay_start_step = lr_state["decay_start_step"]
        if hasattr(learning_rate_callback, "decay_end_step"):
            learning_rate_callback.decay_end_step = lr_state["decay_end_step"]
        if hasattr(learning_rate_callback, "final_lr_fraction"):
            learning_rate_callback.final_lr_fraction = lr_state["final_lr_fraction"]
        if verbose >= 1:
            print(f"恢复学习率衰减回调状态: 当前步数 {lr_state['n_calls']}")

    # 设置信号处理函数以捕获Ctrl+C
    def signal_handler(sig, frame):
        print("\n接收到中断信号(Ctrl+C)，正在保存模型和训练状态...")
        current_steps = save_training_state_on_interrupt_wrapper(
            model=model,
            env=env,
            algorithm=algorithm,
            phase=phase,
            model_dir=model_dir,
            verbose=verbose,
            state_saver_func=save_training_state_on_interrupt,
            callbacks=all_callbacks
        )
        print(f"\n模型在第{current_steps}批次处手动停止训练...")
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    if verbose >= 1:
        print("已设置Ctrl+C中断处理，训练中按Ctrl+C将保存当前状态")

    # 训练模型
    try:
        model.learn(
            total_timesteps=remaining_steps,
            callback=all_callbacks,
            tb_log_name=f"{algorithm}_phase{phase}",
            reset_num_timesteps=not resume
        )
    except EarlyStoppingException as e:
        print(f"早停: {e}")
        pass
    except KeyboardInterrupt:
        print("训练被手动中断(Ctrl+C)，正在保存模型和训练状态...")
        current_steps = save_training_state_on_interrupt_wrapper(
            model=model,
            env=env,
            algorithm=algorithm,
            phase=phase,
            model_dir=model_dir,
            verbose=verbose,
            state_saver_func=save_training_state_on_interrupt,
            callbacks=all_callbacks
        )
        return {"phase_complete": False, "total_steps": current_steps}

    # 保存最终模型
    save_model_and_state(model, env, algorithm, phase, model_dir, suffix="latest", verbose=verbose)

    # 评估当前性能
    phase_complete = eval_callback.get_phase_complete()
    if phase_complete:
        print(f"阶段{phase}完成! 平均奖励: {eval_callback.last_mean_reward:.2f}")
    else:
        print(f"阶段{phase}未完成. 最佳平均奖励: {eval_callback.best_mean_reward:.2f}")

    # 保存最终模型
    final_model_path = os.path.join(model_dir, f"{algorithm}_phase{phase}")
    save_model_and_state(model, env, algorithm, phase, final_model_path, suffix="", verbose=verbose)

    return {"phase_complete": phase_complete, "total_steps": total_timesteps}
