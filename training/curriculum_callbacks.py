# -*- coding: utf-8 -*-
"""
@File    : curriculum_callbacks.py
@Author  : zhangjian
@Desc    : 课程学习的回调函数
"""

import os
import time
import numpy as np
from typing import Dict, Any, Optional
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization
from copy import deepcopy


class CurriculumEvalCallback(EvalCallback):
    """增强版课程学习评估回调"""

    def __init__(
            self,
            eval_env: VecEnv,
            phase: int,
            reward_threshold: float,
            callback_on_new_best: Optional[BaseCallback] = None,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            log_path: Optional[str] = None,
            best_model_save_path: Optional[str] = None,
            deterministic: bool = True,
            render: bool = False,
            verbose: int = 1,
            warn: bool = True,
            early_stop_patience: int = 10,
            min_delta: float = 0.1,
            std_threshold_ratio: float = 0.5,  # 新增参数：标准差阈值比例
    ):
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
        )
        self.last_std_reward = 0  # 初始化为0，将在第一次评估后更新
        self.phase = phase
        self.reward_threshold = reward_threshold
        self.phase_complete = False
        self.std_threshold_ratio = std_threshold_ratio  # 标准差阈值比例

        # 早停相关参数
        self.early_stop_patience = early_stop_patience
        self.min_delta = min_delta
        self.no_improvement_count = 0
        self.best_mean_reward_for_early_stop = -np.inf

        # 评估历史
        self.evaluation_history = []

    def _on_step(self) -> bool:
        """每步训练后的回调处理"""
        # 如果阶段已完成，则不再评估
        if self.phase_complete:
            return False

        # 保存调用父类方法前的评估次数
        previous_eval_count = len(self.evaluations_timesteps) if hasattr(self, 'evaluations_timesteps') else 0

        # 在评估前同步训练和评估环境的归一化统计数据
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # 获取训练环境
            train_env = self.model.get_env()

            # 同步归一化统计数据
            if hasattr(train_env, 'obs_rms') and hasattr(self.eval_env, 'obs_rms'):
                self.eval_env.obs_rms = deepcopy(train_env.obs_rms)
            if hasattr(train_env, 'ret_rms') and hasattr(self.eval_env, 'ret_rms'):
                self.eval_env.ret_rms = deepcopy(train_env.ret_rms)

            if self.verbose > 0:
                print("已同步训练和评估环境的归一化统计数据")

        # 调用父类的评估逻辑
        continue_training = super()._on_step()

        # 检查是否刚完成了一次评估
        current_eval_count = len(self.evaluations_timesteps) if hasattr(self, 'evaluations_timesteps') else 0
        just_evaluated = current_eval_count > previous_eval_count

        # 如果刚完成评估
        if just_evaluated and hasattr(self, 'evaluations_results') and len(self.evaluations_results) > 0:
            # 计算并更新标准差
            latest_rewards = self.evaluations_results[-1]
            self.last_std_reward = float(np.std(latest_rewards))

            # 记录评估历史
            self.evaluation_history.append({
                'timesteps': self.num_timesteps,
                'mean_reward': self.last_mean_reward,
                'std_reward': self.last_std_reward
            })

            if self.verbose > 0:
                print(f"评估结果 - 平均奖励: {self.last_mean_reward:.2f}, 标准差: {self.last_std_reward:.2f}")

            # 检查是否达到阶段完成标准
            if self.reward_threshold is None:
                # 如果没有设置奖励阈值，则继续训练，只打印信息
                if self.verbose > 0:
                    print(f"\n=== 阶段{self.phase}进行中 ===")
                    print(f"当前平均奖励为{self.last_mean_reward:.2f}")
                    print(f"标准差为{self.last_std_reward:.2f}")
                # 返回True表示继续训练
                return True
            elif ((self.last_mean_reward is not None
                   and self.last_mean_reward >= self.reward_threshold)
                  and self.last_std_reward <= self.last_mean_reward * self.std_threshold_ratio):
                self.phase_complete = True
                if self.verbose > 0:
                    print(f"\n=== 阶段{self.phase}完成！===")
                    print(f"平均奖励达到{self.last_mean_reward:.2f}，超过阈值{self.reward_threshold}")
                    print(f"标准差为{self.last_std_reward:.2f}，低于平均值的{self.std_threshold_ratio * 100}%")
                    print(f"在训练步数{self.num_timesteps}时达到目标\n")
                return False  # 停止训练

            # 检查是否应该早停
            if self.early_stop_patience > 0:
                # 如果当前奖励比之前的最佳奖励提高了min_delta，则重置计数器
                if self.last_mean_reward > self.best_mean_reward_for_early_stop + self.min_delta:
                    self.best_mean_reward_for_early_stop = self.last_mean_reward
                    self.no_improvement_count = 0
                    if self.verbose > 0:
                        print(f"发现更好的奖励: {self.last_mean_reward:.2f}")
                else:
                    self.no_improvement_count += 1
                    if self.verbose > 0:
                        print(f"奖励无明显提升: {self.no_improvement_count}/{self.early_stop_patience}")

                # 如果连续early_stop_patience次评估没有提升，则早停
                if self.no_improvement_count >= self.early_stop_patience:
                    if self.verbose > 0:
                        print(f"\n=== 早停触发 ===")
                        print(f"连续{self.early_stop_patience}次评估没有明显提升")
                        print(f"最佳平均奖励: {self.best_mean_reward:.2f}, 阈值: {self.reward_threshold}\n")
                    return False  # 停止训练

        return continue_training

    def get_phase_complete(self) -> bool:
        """返回阶段是否完成"""
        return self.phase_complete

    def get_best_reward(self) -> float:
        """返回最佳平均奖励"""
        return self.best_mean_reward if hasattr(self, 'best_mean_reward') else -np.inf

    def get_evaluation_history(self) -> list:
        """返回评估历史"""
        return self.evaluation_history

    def should_early_stop(self) -> bool:
        """是否应该早停"""
        return self.no_improvement_count >= self.early_stop_patience


class SaveLatestModelCallback(BaseCallback):
    def __init__(self, save_path, save_freq=10000, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path)
            if hasattr(self.training_env, "save"):
                self.training_env.save(f"{self.save_path}_vecnorm.pkl")
        return True


class SaveCheckpointCallback(BaseCallback):
    def __init__(self, algorithm, save_dir, save_freq=50000, max_checkpoints=5, verbose=1):
        super().__init__(verbose)
        self.algorithm = algorithm
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            timestamp = int(time.time())
            checkpoint_path = os.path.join(
                self.save_dir,
                f"checkpoint_{self.algorithm}_{self.num_timesteps}"
            )
            self.model.save(checkpoint_path)
            if hasattr(self.training_env, "save"):
                self.training_env.save(f"{checkpoint_path}_vecnorm.pkl")

            self.checkpoints.append(checkpoint_path)

            # 管理检查点数量
            if len(self.checkpoints) > self.max_checkpoints:
                old_checkpoint = self.checkpoints.pop(0)
                if os.path.exists(f"{old_checkpoint}.zip"):
                    os.remove(f"{old_checkpoint}.zip")
                if os.path.exists(f"{old_checkpoint}_vecnorm.pkl"):
                    os.remove(f"{old_checkpoint}_vecnorm.pkl")

        return True


class EarlyStoppingException(Exception):
    """当训练应该提前停止时抛出的异常"""
    pass


class LinearLRDecayCallback(BaseCallback):
    """
    基于初始学习率的线性衰减回调函数
    """

    def __init__(self, verbose=0, decay_start_step=0, decay_end_step=None, final_lr_fraction=0.1):
        """
        参数:
            verbose: 日志详细程度
            decay_start_step: 开始衰减的步数
            decay_end_step: 结束衰减的步数，如果为None则使用总训练步数
            final_lr_fraction: 最终学习率相对于初始学习率的比例
        """
        super(LinearLRDecayCallback, self).__init__(verbose)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.final_lr_fraction = final_lr_fraction
        self.initial_lr = None

    def _init_callback(self) -> None:
        """初始化回调，获取初始学习率"""
        if self.initial_lr is None:
            self.initial_lr = self.model.learning_rate
        self.current_lr = self.initial_lr
        print(f"初始学习率: {self.initial_lr}")

        # 验证模型优化器结构
        if hasattr(self.model, 'policy'):
            if hasattr(self.model.policy, 'optimizer'):
                print("模型使用单个优化器")
            elif hasattr(self.model.policy, 'actor') and hasattr(self.model.policy.actor, 'optimizer'):
                print("模型使用分离的Actor优化器")
            elif hasattr(self.model.policy, 'critic') and hasattr(self.model.policy.critic, 'optimizer'):
                print("模型使用分离的Critic优化器")
            else:
                print("警告: 无法识别优化器结构")

    def _on_step(self) -> bool:
        """每步更新学习率"""
        # 只在特定步数更新学习率，例如每100步
        if self.decay_start_step <= self.n_calls <= self.decay_end_step and self.n_calls % 1000 == 0:
            # 只有在衰减范围内才调整学习率
            if self.decay_start_step <= self.n_calls <= self.decay_end_step:
                # 计算当前进度
                progress = (self.n_calls - self.decay_start_step) / (self.decay_end_step - self.decay_start_step)
                # 线性衰减：从初始值到 final_lr_fraction * 初始值
                current_lr = self.initial_lr * (1 - (1 - self.final_lr_fraction) * progress)

                # 更新PPO优化器
                if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
                    for param_group in self.model.policy.optimizer.param_groups:
                        param_group['lr'] = current_lr

                # 更新SAC优化器
                if hasattr(self.model, 'actor') and hasattr(self.model.actor, 'optimizer'):
                    for param_group in self.model.actor.optimizer.param_groups:
                        param_group['lr'] = current_lr

                if hasattr(self.model, 'critic') and hasattr(self.model.critic, 'optimizer'):
                    for param_group in self.model.critic.optimizer.param_groups:
                        param_group['lr'] = current_lr

                # 更新SAC温度参数优化器(如果有)
                if hasattr(self.model, 'log_ent_coef') and hasattr(self.model,
                                                                   'ent_coef_optimizer') and self.model.ent_coef_optimizer is not None:
                    for param_group in self.model.ent_coef_optimizer.param_groups:
                        param_group['lr'] = current_lr

                # 更新SB3的日志值（尝试覆盖内部记录的学习率）
                if hasattr(self.model, 'logger') and self.model.logger:
                    self.model.logger.record("train/learning_rate", current_lr)
                if self.model.logger:
                    self.model.logger.record("custom/actual_learning_rate", current_lr)

        return True
