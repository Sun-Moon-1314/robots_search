import os

from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config.maze_search.default_config import LOGS_DIR, MODELS_DIR


def train_sb3(env, algorithm="SAC", total_timesteps=300000, save_freq=10000):
    """
    使用Stable Baselines 3训练迷宫环境

    Args:
        env: 环境实例
        algorithm: 算法名称，"PPO", "SAC", "A2C"之一
        total_timesteps: 总训练步数
        save_freq: 保存模型的频率

    Returns:
        训练好的模型
    """
    print(f"开始使用 {algorithm} 训练迷宫环境...")

    # 包装环境以记录性能
    env = Monitor(env)

    # 将环境向量化并标准化奖励
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)

    # 选择算法
    if algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=LOGS_DIR,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # 增加熵系数，鼓励探索
            policy_kwargs=dict(
                net_arch=[dict(pi=[256, 256], vf=[256, 256])]
            )
        )
    elif algorithm == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=LOGS_DIR,
            learning_rate=3e-4,
            buffer_size=300000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",  # 自动调整熵系数
            learning_starts=5000,  # 增加初始随机探索的步数
            policy_kwargs=dict(
                net_arch=[256, 256]
            )
        )
    elif algorithm == "A2C":
        model = A2C(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=LOGS_DIR,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            ent_coef=0.01,
            policy_kwargs=dict(
                net_arch=[dict(pi=[64, 64], vf=[64, 64])]
            )
        )
    else:
        raise ValueError(f"不支持的算法: {algorithm}")

    # 训练模型
    try:
        # 每save_freq步保存一次模型
        for i in range(1, total_timesteps + 1, save_freq):
            model.learn(total_timesteps=min(save_freq, total_timesteps - i + 1),
                        reset_num_timesteps=False,
                        tb_log_name=f"{algorithm}")

            # 保存模型
            model_path = os.path.join(MODELS_DIR, f"{algorithm}_{i}")
            model.save(model_path)
            # 同时保存归一化的环境状态
            env.save(os.path.join(MODELS_DIR, f"vec_normalize_{algorithm}_{i}.pkl"))
            print(f"模型保存在: {model_path}")

    finally:
        # 保存最终模型
        model_path = os.path.join(MODELS_DIR, f"{algorithm}_final")
        model.save(model_path)
        # 保存归一化的环境状态
        env.save(os.path.join(MODELS_DIR, f"vec_normalize_{algorithm}_final.pkl"))
        print(f"最终模型保存在: {model_path}")

    return model, env
