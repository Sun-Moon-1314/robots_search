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
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization
from path_config import *
from custom_design.custom_common import get_config
from evaluation.evaluator import get_base_env


class CurriculumEvalCallback(EvalCallback):
    """增强版课程学习评估回调，支持奖励阈值和性能指标两种判断方式"""

    def __init__(
            self,
            eval_env: VecEnv,
            phase: int = 1,
            performance_thresholds: Optional[dict] = None,
            callback_on_new_best: Optional[BaseCallback] = None,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            log_path: Optional[str] = None,
            best_model_save_path: Optional[str] = None,
            deterministic: bool = True,
            render: bool = False,
            verbose: int = 1,
            warn: bool = True,
            early_stop_patience: int = 5,
            min_delta: float = 0.1,
            std_threshold_ratio: float = 0.5,
            distance_threshold: float = 0.5,
            tilt_threshold: float = 0.3,
            check_direction: bool = False,
            success_rate_threshold: float = 0.9,
            success_window_size: int = 5,
            eval_window_size: int = 100,  # 新增参数：评估窗口大小，用于平均值计算
            max_steps_threshold: int = 70,  # 新增参数：成功任务的最大步数阈值
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
        self.phase = phase
        self.performance_thresholds = performance_thresholds
        self.phase_complete = False
        self.std_threshold_ratio = std_threshold_ratio

        # 早停相关参数
        self.early_stop_patience = early_stop_patience
        self.min_delta = min_delta
        self.no_improvement_count = 0
        self.best_mean_reward_for_early_stop = -np.inf

        # 状态条件早停参数
        self.distance_threshold = distance_threshold
        self.tilt_threshold = tilt_threshold
        self.check_direction = check_direction

        # 成功率早停参数
        self.success_rate_threshold = success_rate_threshold
        self.success_window_size = success_window_size
        self.success_history = []
        self.success_rate = 0.0
        self.success_early_stop_count = 0
        # 评估窗口参数
        self.eval_window_size = eval_window_size  # 评估窗口大小，用于平均值计算
        # 新增：成功任务最大步数阈值
        self.max_steps_threshold = max_steps_threshold
        self.adjusted_max_steps_threshold = max_steps_threshold  # 动态调整后的阈值
        # 评估历史
        self.evaluation_history = []
        self.performance_metrics = {}
        self.performance_history = []

    def _check_state_conditions(self) -> bool:
        """检查是否满足状态条件以触发早停"""
        if not self.performance_metrics:
            return False

        distance_ok = (
                'distance_to_target' in self.performance_metrics and
                self.performance_metrics['distance_to_target'] <= self.distance_threshold
        )

        tilt_ok = (
                'tilt_angle' in self.performance_metrics and
                self.performance_metrics['tilt_angle'] <= self.tilt_threshold
        )

        direction_ok = True
        if self.check_direction:
            direction_ok = (
                    'movement_alignment' in self.performance_metrics and
                    self.performance_metrics['movement_alignment'] >= 0.5
            )

        steps_ok = (
                'current_step' in self.performance_metrics and
                self.performance_metrics['current_step'] <= self.adjusted_max_steps_threshold
        )

        all_conditions_met = distance_ok and tilt_ok and direction_ok and steps_ok

        if self.verbose > 1:
            print(f"早停状态条件检查：")
            print(
                f"  距离条件: {distance_ok} ({self.performance_metrics.get('distance_to_target', 'N/A'):.2f} <= {self.distance_threshold:.2f})")
            print(
                f"  倾斜角条件: {tilt_ok} ({self.performance_metrics.get('tilt_angle', 'N/A'):.2f} <= {self.tilt_threshold:.2f})")
            if self.check_direction:
                print(
                    f"  方向条件: {direction_ok} ({self.performance_metrics.get('movement_alignment', 'N/A'):.2f} >= 0.5)")
            print(
                f"  步数条件: {steps_ok} ({self.performance_metrics.get('current_step', 'N/A'):.2f} <= {self.adjusted_max_steps_threshold:.2f})")

        return all_conditions_met

    def _check_success_rate_condition(self) -> bool:
        """检查是否满足成功率条件以触发早停，基于窗口内平均值"""
        if len(self.success_history) < self.eval_window_size:
            if self.verbose > 1:
                print(f"评估次数不足 {self.eval_window_size}，当前为 {len(self.success_history)}，暂不触发早停")
            return False

        # 获取窗口内的成功率历史记录
        window_success_history = self.success_history[-self.eval_window_size:]
        total_episodes = len(window_success_history)
        successful_episodes = sum(window_success_history)
        self.success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0.0

        success_condition_met = self.success_rate >= self.success_rate_threshold

        if success_condition_met:
            self.success_early_stop_count += 1
            if self.verbose > 0:
                print(f"成功率条件满足: {self.success_rate:.2%} >= {self.success_rate_threshold:.2%}, "
                      f"连续满足次数: {self.success_early_stop_count}")
        else:
            self.success_early_stop_count = 0
            if self.verbose > 0:
                print(f"成功率条件未满足: {self.success_rate:.2%} < {self.success_rate_threshold:.2%}, 重置计数器")

        return self.success_early_stop_count >= 3  # 仍然保留连续 3 次的条件，可以根据需求调整

    def _custom_evaluate(self):
        """自定义评估方法，收集性能指标和成功率"""
        # 动态调整 max_steps_threshold，检查环境中是否有 steps_per_grid 属性
        if hasattr(self.training_env, 'steps_per_grid'):
            self.adjusted_max_steps_threshold = min(self.max_steps_threshold, self.training_env.steps_per_grid)
            if self.verbose > 0:
                print(
                    f"动态调整步数阈值: 使用 {self.adjusted_max_steps_threshold} (取 {self.max_steps_threshold} 和环境 steps_per_grid {self.training_env.steps_per_grid} 的最小值)")
        else:
            self.adjusted_max_steps_threshold = self.max_steps_threshold
            if self.verbose > 0:
                print(f"环境中无 steps_per_grid 属性，使用默认步数阈值: {self.adjusted_max_steps_threshold}")

        episode_performance_metrics = {
            "phase": None,
            "current_step": [],
            "tilt_angle": [],
            "distance_to_target": [],
            "movement_alignment": [],
            "direction_exploration": []
        }
        episode_rewards, episode_lengths = [], []
        episode_successes = []
        eval_env = self.training_env
        eval_env.reset()
        if hasattr(eval_env, 'prev_angle_diff'):
            eval_env.prev_angle_diff = None

        for i in range(self.n_eval_episodes):
            obs = eval_env.reset()
            if hasattr(eval_env, 'prev_angle_diff'):
                eval_env.prev_angle_diff = None

            done, state = False, None
            episode_reward = 0.0
            episode_length = 0
            episode_metrics = {
                "current_step": 0,
                "tilt_angle": [],
                "distance_to_target": [],
                "movement_alignment": [],
                "direction_exploration": []
            }
            episode_success = False

            while not done:
                action, state = self.model.predict(obs, state=state, deterministic=self.deterministic)
                obs, reward, done, info = eval_env.step(action)

                # 收集性能指标
                if hasattr(eval_env, "get_performance_metrics"):
                    metrics = eval_env.get_performance_metrics()
                    for key, value in metrics.items():
                        if key in episode_metrics:
                            if isinstance(value, (list, tuple)):
                                episode_metrics[key].extend(value)
                            else:
                                episode_metrics[key].append(value)

                # 手动提取关键指标 - 适配28维观察空间，处理批量数据
                current_obs = obs[0] if len(obs.shape) > 1 and obs.shape[0] == 1 else obs
                if len(current_obs) >= 28:
                    roll, pitch = current_obs[5], current_obs[6]
                    tilt_angle = np.sqrt(roll ** 2 + pitch ** 2)
                    episode_metrics["tilt_angle"].append(tilt_angle)
                    dx, dy = current_obs[3], current_obs[4]
                    distance = np.sqrt(dx ** 2 + dy ** 2)
                    episode_metrics["distance_to_target"].append(distance)
                    movement_alignment = current_obs[27]
                    episode_metrics["movement_alignment"].append(movement_alignment)
                    yaw = current_obs[2]
                    target_angle = np.arctan2(dy, dx)
                    angle_diff = np.arctan2(np.sin(target_angle - yaw), np.cos(target_angle - yaw))
                    if hasattr(eval_env, 'prev_angle_diff') and eval_env.prev_angle_diff is not None:
                        angle_diff_change = abs(angle_diff) - abs(eval_env.prev_angle_diff)
                        episode_metrics["direction_exploration"].append(-angle_diff_change)
                    if hasattr(eval_env, 'prev_angle_diff'):
                        eval_env.prev_angle_diff = angle_diff

                episode_reward += reward[0] if isinstance(reward, (list, tuple, np.ndarray)) else reward
                episode_length += 1
                if isinstance(done, (list, tuple, np.ndarray)):
                    done = any(done)

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # 回合结束后，基于平均性能指标判断是否成功
            avg_metrics = {}
            for key, values in episode_metrics.items():
                if values:
                    avg_metrics[key] = np.mean(values)

            # 成功条件：距离、倾斜角、可选的方向对齐以及步数均满足阈值
            distance_ok = 'distance_to_target' in avg_metrics and avg_metrics[
                'distance_to_target'] <= self.distance_threshold
            tilt_ok = 'tilt_angle' in avg_metrics and avg_metrics['tilt_angle'] <= self.tilt_threshold
            direction_ok = True
            if self.check_direction:
                direction_ok = 'movement_alignment' in avg_metrics and avg_metrics['movement_alignment'] >= 0.5
            steps_ok = episode_length <= self.adjusted_max_steps_threshold  # 新增步数条件

            episode_success = distance_ok and tilt_ok and direction_ok and steps_ok
            episode_successes.append(episode_success)

            if self.verbose > 1:
                print(f"回合 {i + 1}/{self.n_eval_episodes} 成功判断：")
                print(
                    f"  距离条件: {distance_ok} ({avg_metrics.get('distance_to_target', 'N/A'):.2f} <= {self.distance_threshold:.2f})")
                print(
                    f"  倾斜角条件: {tilt_ok} ({avg_metrics.get('tilt_angle', 'N/A'):.2f} <= {self.tilt_threshold:.2f})")
                if self.check_direction:
                    print(f"  方向条件: {direction_ok} ({avg_metrics.get('movement_alignment', 'N/A'):.2f} >= 0.5)")
                print(f"  步数条件: {steps_ok} ({episode_length} <= {self.adjusted_max_steps_threshold})")
                print(f"  是否成功: {episode_success}")

            episode_performance_metrics["phase"] = self.phase
            for key, values in episode_metrics.items():
                if key == "current_step":
                    episode_performance_metrics[key].append(episode_length)
                elif values:
                    if key in ["tilt_angle", "distance_to_target"]:
                        episode_performance_metrics[key].append(np.mean(values))
                    elif key == "movement_alignment":
                        episode_performance_metrics[key].append(np.mean(values))
                    elif key == "direction_exploration":
                        episode_performance_metrics[key].append(np.mean(values) if values else 0)

            # 更新性能指标
            self.performance_metrics = {}
            for key, values in episode_performance_metrics.items():
                if values and key != "phase":
                    self.performance_metrics[key] = np.mean(values)

            # 更新成功率历史记录
            self.success_history.extend(episode_successes)
            if len(self.success_history) > self.eval_window_size:
                self.success_history = self.success_history[-self.eval_window_size:]

            history_entry = {
                "phase": self.phase,
                'timesteps': self.num_timesteps,
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'success_rate': sum(episode_successes) / len(episode_successes) if episode_successes else 0.0
            }
            history_entry.update(self.performance_metrics)
            self.evaluation_history.append(history_entry)

            if not hasattr(self, "evaluations_results"):
                self.evaluations_results = []
            if not hasattr(self, "evaluations_timesteps"):
                self.evaluations_timesteps = []

            self.performance_history.append(self.performance_metrics.copy())
            self.evaluations_results.append(episode_rewards)
            self.evaluations_timesteps.append(self.num_timesteps)

            if self.verbose > 0:
                print(f"评估结果 - 平均奖励: {np.mean(episode_rewards):.2f}, 标准差: {np.std(episode_rewards):.2f}")
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in self.performance_metrics.items()])
                print(f"性能指标: {metrics_str}")
                success_rate = sum(episode_successes) / len(episode_successes) if episode_successes else 0.0
                print(f"成功率: {success_rate:.2%} ({sum(episode_successes)}/{len(episode_successes)} 回合成功)")

    def _on_step(self) -> bool:
        """
        在每个训练步骤后调用，检查是否需要进行评估。
        返回 False 将停止训练。
        """
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            # 进行评估
            self._custom_evaluate()

            # 检查是否达到性能阈值
            if self.performance_thresholds:
                all_thresholds_met = True
                for metric, threshold in self.performance_thresholds.items():
                    if metric not in self.performance_metrics:
                        all_thresholds_met = False
                        if self.verbose > 0:
                            print(f"性能指标 {metric} 未找到，无法检查阈值")
                        break
                    current_value = self.performance_metrics[metric]
                    if metric in ["distance_to_target", "tilt_angle", "current_step"]:
                        if current_value > threshold:
                            all_thresholds_met = False
                            if self.verbose > 0:
                                print(f"性能指标 {metric} 未达到阈值: {current_value:.2f} > {threshold:.2f}")
                    else:
                        if current_value < threshold:
                            all_thresholds_met = False
                            if self.verbose > 0:
                                print(f"性能指标 {metric} 未达到阈值: {current_value:.2f} < {threshold:.2f}")

                if all_thresholds_met:
                    if self.verbose > 0:
                        print(f"阶段 {self.phase} 所有性能阈值已达到，标记阶段完成")
                    self.phase_complete = True

            # 检查早停条件 - 基于奖励改进
            mean_reward = self.evaluation_history[-1]['mean_reward']
            if mean_reward > self.best_mean_reward_for_early_stop + self.min_delta:
                self.best_mean_reward_for_early_stop = mean_reward
                self.no_improvement_count = 0
                if self.verbose > 1:
                    print(f"奖励改进: {mean_reward:.2f} > {self.best_mean_reward_for_early_stop:.2f}，重置早停计数器")
            else:
                self.no_improvement_count += 1
                if self.verbose > 1:
                    print(f"奖励无改进，当前计数: {self.no_improvement_count}/{self.early_stop_patience}")

            # 检查成功率条件早停（优先）
            success_rate_met = self._check_success_rate_condition()

            if success_rate_met:
                if self.verbose > 0:
                    print(f"早停触发: 成功率条件连续满足 {self.success_early_stop_count} 次 (>= 3)")
                self.phase_complete = True
                return False

            # 如果成功率未满足，再检查其他条件
            state_conditions_met = self._check_state_conditions()
            if state_conditions_met:
                if self.verbose > 0:
                    print(f"早停触发: 状态条件满足 (距离, 倾斜角等指标达到阈值)")
                return False

            if self.no_improvement_count >= self.early_stop_patience:
                if self.verbose > 0:
                    print(
                        f"早停触发: 奖励连续 {self.no_improvement_count} 次评估无改进 (>= {self.early_stop_patience})")
                return False

            # 检查标准差是否满足条件 (奖励稳定性)
            if len(self.evaluation_history) > 1:
                current_std = self.evaluation_history[-1]['std_reward']
                current_mean = self.evaluation_history[-1]['mean_reward']
                if abs(current_mean) > 1e-6:
                    std_ratio = current_std / abs(current_mean)
                    if std_ratio <= self.std_threshold_ratio:
                        if self.verbose > 0:
                            print(
                                f"奖励稳定性达到阈值: 标准差/平均值 = {std_ratio:.2f} <= {self.std_threshold_ratio:.2f}")
                        self.phase_complete = True
                    else:
                        if self.verbose > 1:
                            print(
                                f"奖励稳定性未达到阈值: 标准差/平均值 = {std_ratio:.2f} > {self.std_threshold_ratio:.2f}")

        return True

    def get_phase_complete(self) -> bool:
        """检查当前阶段是否完成"""
        return self.phase_complete

    def get_evaluation_history(self) -> list:
        """获取评估历史记录"""
        return self.evaluation_history

    def get_performance_metrics(self) -> dict:
        """获取最新的性能指标"""
        return self.performance_metrics


class SaveModelLogCallback(BaseCallback):
    """在保存最佳模型时打印详细日志的回调"""

    def __init__(self, verbose=1):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        """这个方法在每次调用时执行"""
        print(f"\n{'=' * 50}")
        print(f"🌟 发现新的最佳模型！在步数 {self.num_timesteps} 处保存")
        print(f"{'=' * 50}\n")
        return True


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
    def __init__(self, algorithm, save_dir, save_freq=20000, max_checkpoints=5, verbose=1):
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

    def __init__(self, verbose=0, decay_start_step=0, decay_end_step=None, final_lr_fraction=0.1,
                 decay_by_time_steps=10000):
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
        self.decay_by_time_steps = decay_by_time_steps
        self.initial_lr = None

    def _init_callback(self) -> None:
        """初始化回调，获取初始学习率"""
        if self.initial_lr is None:
            self.initial_lr = self.model.learning_rate
        self.current_lr = self.initial_lr

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
        # 只在特定步数更新学习率，例如每1000步
        if self.decay_start_step <= self.n_calls <= self.decay_end_step and self.n_calls % self.decay_by_time_steps == 0:
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

                # 同步学习率到模型属性
                if hasattr(self.model, 'learning_rate'):
                    self.model.learning_rate = current_lr

                # 更新日志值
                if hasattr(self.model, 'logger') and self.model.logger:
                    self.model.logger.record("train/learning_rate", current_lr)
                if self.model.logger:
                    self.model.logger.record("custom/actual_learning_rate", current_lr)

                self.current_lr = current_lr  # 保存当前学习率到回调实例中

        return True


class BallPositionCallback(BaseCallback):
    """
    简化版本的回调函数，只记录小球位置分布并计算相关指标，不进行可视化
    """

    def __init__(self, eval_env, verbose=0):
        super(BallPositionCallback, self).__init__(verbose)
        self.eval_env = get_base_env(eval_env)  # 评估环境
        self.ball_positions = []  # 存储所有观察到的球位置
        self.position_counts = {}  # 计数每个位置出现的次数
        self.last_position = None  # 记录上一次的位置

        # 尝试获取迷宫大小或设置默认值
        if hasattr(self.eval_env, 'maze_size'):
            self.maze_size = self.eval_env.maze_size
        else:
            self.maze_size = (7, 7)  # 默认值

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        """在每次收集新的rollout开始时调用"""
        # 重置环境并记录球的位置
        self.eval_env.reset()

        # 获取球的位置
        ball_pos = self._get_ball_position()

        if ball_pos is not None:
            # 检查位置是否有变化
            position_changed = (self.last_position != ball_pos)
            if position_changed:
                print(f"球位置变化: {self.last_position} -> {ball_pos}")

            # 更新上一次位置
            self.last_position = ball_pos

            # 记录位置
            self.ball_positions.append(ball_pos)

            # 更新位置计数
            pos_key = str(ball_pos)
            if pos_key in self.position_counts:
                self.position_counts[pos_key] += 1
            else:
                self.position_counts[pos_key] = 1

            # 每10次rollout计算并记录指标
            if len(self.ball_positions) % 10 == 0:
                self._calculate_metrics()

    def _get_ball_position(self):
        """尝试获取球的位置"""
        if hasattr(self.eval_env, 'ball_pos'):
            return self.eval_env.ball_pos
        elif hasattr(self.eval_env, 'get_ball_position'):
            return self.eval_env.get_ball_position()
        elif hasattr(self.eval_env, 'observation') and len(self.eval_env.observation) >= 5:
            # 从28维观察空间中提取球的相对位置
            dx, dy = self.eval_env.observation[3], self.eval_env.observation[4]
            robot_x, robot_y = self.eval_env.observation[0], self.eval_env.observation[1]
            # 计算球的绝对位置
            return (robot_x + dx, robot_y + dy)
        return None

    def _calculate_metrics(self):
        """计算并记录位置分布指标"""
        # 创建分布矩阵
        distribution = np.zeros(self.maze_size)

        # 填充分布数据
        for pos in self.ball_positions:
            x, y = pos
            if 0 <= x < self.maze_size[0] and 0 <= y < self.maze_size[1]:
                distribution[x, y] += 1

        # 归一化分布
        if np.sum(distribution) > 0:
            distribution = distribution / np.sum(distribution)

        # 计算熵以量化随机性
        entropy = self._calculate_entropy(distribution)
        self.logger.record('environment/position_entropy', entropy)
        print(f"位置熵: {entropy:.4f}")

        # 计算覆盖率 - 有多少可能的位置被使用了
        total_positions = self.maze_size[0] * self.maze_size[1]
        used_positions = len(self.position_counts)
        coverage = used_positions / total_positions
        self.logger.record('environment/position_coverage', coverage)
        print(f"位置覆盖率: {coverage:.4f} ({used_positions}/{total_positions})")

        # 打印当前记录的唯一位置数
        print(f"已记录 {len(self.ball_positions)} 个位置样本，包含 {used_positions} 个唯一位置")

    def _calculate_entropy(self, distribution):
        """计算分布的熵，量化随机性"""
        # 将零概率替换为很小的数，避免log(0)
        distribution = distribution.flatten()
        distribution = distribution[distribution > 0]
        if len(distribution) == 0:
            return 0
        return -np.sum(distribution * np.log2(distribution))


class RewardMetricsCallback(BaseCallback):
    """
    增强版指标记录回调函数，适配28维观察空间，记录关键指标并转换为易于理解的单位：
    - 最终到达球的距离 (米)
    - 倾斜角度 (度)
    - 偏航角 (度，0-360范围)
    - 完成步数
    - 小球初始位置 (x, y坐标)
    - 目标跟踪指标 (小球在视野中的位置)
    - 机器人正面与球的角度差异
    - 机器人速度和平衡状态
    """

    def __init__(self,
                 verbose=1,
                 log_dir="path/to/tensorboard_logs",
                 log_freq=1000,  # 降低记录频率
                 check_direction=False):
        """
        初始化回调函数

        参数:
            verbose (int): 日志详细程度
            log_dir (str): TensorBoard日志保存目录
            log_freq (int): 记录日志的频率(步数)
            check_direction (bool): 是否检查方向对齐作为成功条件
        """
        super(RewardMetricsCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.check_direction = check_direction

        # 初始化统计数据
        self.episode_count = 0
        self.current_episode_length = 0
        self.episode_log_freq = 100

        # 成功率统计
        self.total_attempts = 0
        self.total_successes = 0
        self.success_history = []  # 记录每个回合是否成功
        self.success_window_size = 100  # 成功率计算窗口大小
        self.success_rate = 0.0

        # 当前步数
        self.current_steps = 0
        # 当前episode的关键指标
        self.current_distance = None
        self.current_tilt = None
        self.current_yaw = None
        self.current_angle_diff = None
        self.prev_angle_diff = None

        # 添加目标跟踪指标
        self.current_target_tracking = None
        self.current_laser_target_position = None
        self.current_front_angle_diff = None  # 机器人正面与球的角度差异

        # 添加速度和平衡状态指标
        self.current_forward_speed = None
        self.current_lateral_speed = None
        self.current_movement_consistency = None
        self.current_roll = None
        self.current_pitch = None
        self.current_roll_rate = None
        self.current_pitch_rate = None

        # 小球位置 - 现在保存初始位置
        self.initial_ball_position = None
        self.current_ball_position = None

        # 当前回合的指标列表，用于计算平均值
        self.episode_distances = []
        self.episode_tilts = []
        self.episode_movement_consistencies = []

        # 定义倾斜状态描述
        self.tilt_descriptions = {
            (0, 5): "非常稳定",
            (5, 10): "稳定",
            (10, 15): "轻微倾斜",
            (15, 25): "明显倾斜",
            (25, 35): "严重倾斜",
            (35, float('inf')): "即将摔倒"
        }

        # 定义目标跟踪质量描述
        self.tracking_descriptions = {
            (0.8, 1.0): "完美居中",
            (0.6, 0.8): "良好居中",
            (0.4, 0.6): "部分居中",
            (0.2, 0.4): "边缘可见",
            (0.0, 0.2): "几乎不可见"
        }

        # 定义移动一致性描述
        self.consistency_descriptions = {
            (0.8, 1.0): "高度一致",
            (0.5, 0.8): "较为一致",
            (0.2, 0.5): "部分一致",
            (0.0, 0.2): "略有一致",
            (-0.2, 0.0): "略微不一致",
            (-0.5, -0.2): "部分不一致",
            (-0.8, -0.5): "较为不一致",
            (-1.0, -0.8): "高度不一致"
        }

        # 成功条件阈值，将在 _on_step 中动态更新
        self.success_distance_threshold = 0.5
        self.success_tilt_threshold = 0.3
        self.max_steps_threshold = 70
        self.adjusted_max_steps_threshold = self.max_steps_threshold

    def _on_step(self) -> bool:
        """每步调用的方法"""
        # 获取当前环境
        env_ = self.training_env
        env = get_base_env(env_)

        # 获取当前课程阶段
        current_phase = getattr(env, 'curriculum_phase', 1)

        # 动态加载配置和成功条件阈值
        try:
            env_config = getattr(env, 'env_config', None)
            if env_config and 'performance_thresholds' in env_config:
                perf_thresholds = env_config['performance_thresholds']
                self.success_distance_threshold = perf_thresholds.get('distance_to_target', 0.5)
                self.success_tilt_threshold = perf_thresholds.get('tilt_angle', 0.3)
                self.max_steps_threshold = perf_thresholds.get('max_steps', 70)
            else:
                # 如果环境中没有配置，回退到默认值或配置文件
                config = get_config(current_phase)
                if 'performance_thresholds' in config:
                    perf_thresholds = config['performance_thresholds']
                    self.success_distance_threshold = perf_thresholds.get('distance_to_target', 0.5)
                    self.success_tilt_threshold = perf_thresholds.get('tilt_angle', 0.3)
                    self.max_steps_threshold = perf_thresholds.get('max_steps', 70)
        except Exception as e:
            if self.verbose > 0:
                print(f"加载配置失败，使用默认阈值: {e}")

        # 获取当前观察值
        if hasattr(env, 'get_raw_obs'):
            obs = env.get_raw_obs()  # 获取原始观察值
            obs = np.round(obs, decimals=6)
        else:
            # 如果环境不支持 get_raw_obs，回退到默认逻辑
            obs_normalized = self.locals.get('new_obs')[0]
            if self.verbose > 0:
                print(f"警告：环境不支持 get_raw_obs，使用归一化数据: {obs_normalized}")
            if hasattr(self.training_env, 'obs_rms'):
                mean = self.training_env.obs_rms.mean
                var = self.training_env.obs_rms.var
                obs_ = obs_normalized * np.sqrt(var + 1e-8) + mean
                _obs = obs_.copy()
                _obs[17:25] = np.round(obs_normalized[17:25]).astype(float)
                obs = np.round(_obs, decimals=6)
            else:
                obs = obs_normalized

        self.current_episode_length += 1
        self.current_steps = self.current_episode_length
        # 尝试从环境中获取步数信息
        try:
            if hasattr(env, 'get_performance_metrics'):
                metrics = env.get_performance_metrics()
                if 'max_steps' in metrics:
                    self.current_steps = metrics.get('current_step', self.current_steps)
            elif hasattr(env, 'current_step'):
                self.current_steps = env.current_step
        except Exception as e:
            if self.verbose > 0:
                print(f"获取步数信息失败: {e}，使用默认步数 {self.current_steps}")

        # 提取关键指标 - 适配28维观察空间
        if len(obs) == 28:  # 确保观察空间维度正确
            # 提取机器人位置和朝向 (索引0-2)
            robot_x, robot_y = obs[0], obs[1]
            yaw = obs[2]
            self.current_yaw = yaw

            # 提取球的相对位置 (索引3-4)
            dx, dy = obs[3], obs[4]
            self.current_distance = np.sqrt(dx ** 2 + dy ** 2)  # 计算到球的距离
            self.current_ball_position = (dx, dy)
            self.episode_distances.append(self.current_distance)  # 记录到回合列表中

            # 提取平衡状态 (索引5-8)
            roll, pitch = obs[5], obs[6]
            roll_rate, pitch_rate = obs[7], obs[8]
            self.current_roll = roll
            self.current_pitch = pitch
            self.current_roll_rate = roll_rate
            self.current_pitch_rate = pitch_rate
            self.current_tilt = np.sqrt(roll ** 2 + pitch ** 2)  # 计算倾斜角度
            self.episode_tilts.append(self.current_tilt)  # 记录到回合列表中

            # 计算机器人朝向与目标方向的角度差
            target_angle = np.arctan2(dy, dx)
            angle_diff = np.arctan2(np.sin(target_angle - yaw), np.cos(target_angle - yaw))
            self.current_angle_diff = angle_diff

            # 更新上一步角度差
            self.prev_angle_diff = angle_diff

            # 如果是新的episode的第一步，记录初始位置
            if self.current_episode_length == 1:
                self.initial_ball_position = self.current_ball_position

            # 提取激光传感器数据 (索引9-16)和目标类型 (索引17-24)
            laser_distances = obs[9:17]  # 8个激光数据
            laser_targets = obs[17:25]  # 8个激光检测目标类型
            # 寻找目标类型为2的索引（表示小球）
            ball_indices = [i for i, target in enumerate(laser_targets) if int(target) == 2]
            if len(ball_indices) > 0:
                # 计算目标跟踪指标
                center_index = 3.5  # 激光传感器阵列的中心位置（0-7索引系统）
                avg_deviation = sum(abs(idx - center_index) for idx in ball_indices) / len(ball_indices)
                normalized_deviation = avg_deviation / 3.5
                self.current_target_tracking = 1.0 - normalized_deviation

                # 记录小球在激光传感器中的平均位置
                avg_position = sum(ball_indices) / len(ball_indices)
                self.current_laser_target_position = avg_position

                # 计算机器人正面与球的角度差异
                laser_fov = np.pi  # 180度视野范围（弧度）
                position_ratio = (avg_position - center_index) / 3.5  # 范围[-1, 1]
                front_angle_diff = position_ratio * (laser_fov / 2)  # 转换为角度差异
                self.current_front_angle_diff = front_angle_diff
            else:
                self.current_target_tracking = 0.0
                self.current_laser_target_position = None
                self.current_front_angle_diff = None

            # 提取速度和移动一致性 (索引25-27)
            self.current_forward_speed = obs[25]  # 前进速度
            self.current_lateral_speed = obs[26]  # 侧向速度
            self.current_movement_consistency = obs[27]  # 移动一致性
            self.episode_movement_consistencies.append(self.current_movement_consistency)  # 记录到回合列表中

        # 检查episode是否结束
        done = env.get_done()
        # 每隔 log_freq 步记录一次数据到 TensorBoard，且确保不是 episode 的第一步
        if (self.log_freq is not None
                and self.num_timesteps % self.log_freq == 0 and self.current_distance is not None
                and self.current_distance < 1.0):
            # 检查关键指标中为 0 的数量
            zero_count = 0
            key_metrics = [
                self.current_distance if self.current_distance is not None else 0,
                self.current_tilt if self.current_tilt is not None else 0,
                self.current_yaw if self.current_yaw is not None else 0,
                self.current_angle_diff if self.current_angle_diff is not None else 0,
                self.current_target_tracking if self.current_target_tracking is not None else 0,
                self.current_forward_speed if self.current_forward_speed is not None else 0,
                self.current_lateral_speed if self.current_lateral_speed is not None else 0,
                self.current_movement_consistency if self.current_movement_consistency is not None else 0
            ]
            for metric in key_metrics:
                if abs(metric) < 1e-6:  # 考虑到浮点数精度问题，使用一个很小的阈值来判断是否为 0
                    zero_count += 1

            # 如果为 0 的指标数量超过 5 个，则不记录数据
            if zero_count > 5:
                return True  # 跳过记录，直接返回

            # 记录方向探索进展
            if self.prev_angle_diff is not None:
                angle_diff_change = abs(angle_diff) - abs(self.prev_angle_diff)
                self.logger.record("metrics/angle_diff_change", angle_diff_change)
            # 记录当前步骤的关键指标
            if self.current_distance is not None:
                self.logger.record("metrics/distance_to_ball", self.current_distance)
            if self.current_tilt is not None:
                tilt_degrees = np.degrees(self.current_tilt)
                self.logger.record("metrics/tilt_angle_deg", tilt_degrees)
            if self.current_yaw is not None:
                yaw_degrees = np.degrees(self.current_yaw) % 360
                self.logger.record("metrics/yaw_angle_deg", yaw_degrees)
            if self.current_angle_diff is not None:
                angle_diff_degrees = np.degrees(self.current_angle_diff)
                self.logger.record("metrics/angle_diff_deg", angle_diff_degrees)
                self.logger.record("metrics/abs_angle_diff_deg", abs(angle_diff_degrees))

            # 记录目标跟踪指标
            if self.current_target_tracking is not None:
                self.logger.record("metrics/target_tracking_score", self.current_target_tracking)
            if self.current_laser_target_position is not None:
                self.logger.record("metrics/laser_target_position", self.current_laser_target_position)
            if self.current_front_angle_diff is not None:
                front_angle_diff_degrees = np.degrees(self.current_front_angle_diff)
                self.logger.record("metrics/front_angle_diff_deg", front_angle_diff_degrees)
                self.logger.record("metrics/abs_front_angle_diff_deg", abs(front_angle_diff_degrees))

            # 记录小球当前位置
            if self.current_ball_position is not None:
                self.logger.record("metrics/ball_position_x", self.current_ball_position[0])
                self.logger.record("metrics/ball_position_y", self.current_ball_position[1])

            # 记录平衡状态
            if self.current_roll is not None:
                self.logger.record("metrics/roll_deg", np.degrees(self.current_roll))
            if self.current_pitch is not None:
                self.logger.record("metrics/pitch_deg", np.degrees(self.current_pitch))

            # 记录速度和移动一致性
            if self.current_forward_speed is not None:
                self.logger.record("metrics/forward_speed", self.current_forward_speed)
            if self.current_lateral_speed is not None:
                self.logger.record("metrics/lateral_speed", self.current_lateral_speed)
            if self.current_movement_consistency is not None:
                self.logger.record("metrics/movement_consistency", self.current_movement_consistency)

            # 记录当前步数
            self.logger.record("metrics/current_steps", self.current_steps)
            # 记录成功率
            self.logger.record("metrics/success_rate", self.success_rate)
            # 写入数据到 TensorBoard
            self.logger.dump(self.num_timesteps)

        # 只有在 episode 结束时才重置相关变量并判断成功率
        if done:
            self.total_attempts += 1
            self.episode_count += 1

            # 动态调整 max_steps_threshold，检查环境中是否有 steps_per_grid 属性
            if hasattr(env, 'steps_per_grid'):
                self.adjusted_max_steps_threshold = min(self.max_steps_threshold, env.steps_per_grid)
                if self.verbose > 0 and self.episode_count % self.episode_log_freq == 0:
                    print(
                        f"动态调整步数阈值: 使用 {self.adjusted_max_steps_threshold} (取 {self.max_steps_threshold} 和环境 steps_per_grid {env.steps_per_grid} 的最小值)")
            else:
                self.adjusted_max_steps_threshold = self.max_steps_threshold
                if self.verbose > 0 and self.episode_count % self.episode_log_freq == 0:
                    print(f"环境中无 steps_per_grid 属性，使用默认步数阈值: {self.adjusted_max_steps_threshold}")

            # 计算回合的平均指标
            avg_distance = np.mean(self.episode_distances) if self.episode_distances else float('inf')
            avg_tilt = np.mean(self.episode_tilts) if self.episode_tilts else float('inf')
            avg_movement_consistency = np.mean(
                self.episode_movement_consistencies) if self.episode_movement_consistencies else 0.0

            # 判断是否成功
            distance_ok = avg_distance <= self.success_distance_threshold
            tilt_ok = avg_tilt <= self.success_tilt_threshold
            direction_ok = True
            if self.check_direction:
                direction_ok = avg_movement_consistency >= 0.5
            steps_ok = self.current_steps <= self.adjusted_max_steps_threshold  # 新增步数条件

            episode_success = distance_ok and tilt_ok and direction_ok and steps_ok
            if episode_success:
                self.total_successes += 1

            # 更新成功历史记录
            self.success_history.append(episode_success)
            if len(self.success_history) > self.success_window_size:
                self.success_history = self.success_history[-self.success_window_size:]

            # 计算成功率
            total_episodes = len(self.success_history)
            successful_episodes = sum(self.success_history)
            self.success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0.0

            # 记录成功率和判断结果
            if self.verbose > 0 and self.episode_count % self.episode_log_freq == 0:
                print(
                    f"回合 {self.episode_count} 结束，成功率: {self.success_rate:.2%} ({successful_episodes}/{total_episodes})")
                print(f"成功条件检查：")
                print(f"  距离条件: {distance_ok} ({avg_distance:.2f} <= {self.success_distance_threshold:.2f})")
                print(f"  倾斜角条件: {tilt_ok} ({avg_tilt:.2f} <= {self.success_tilt_threshold:.2f})")
                if self.check_direction:
                    print(f"  方向条件: {direction_ok} ({avg_movement_consistency:.2f} >= 0.5)")
                print(f"  步数条件: {steps_ok} ({self.current_steps} <= {self.adjusted_max_steps_threshold})")
                print(f"  是否成功: {episode_success}")

            # 重置回合相关变量
            self.current_episode_length = 0
            self.initial_ball_position = None  # 重置初始位置，为下一个 episode 做准备
            self.prev_angle_diff = None  # 重置上一步角度差
            self.episode_distances = []  # 重置回合指标列表
            self.episode_tilts = []
            self.episode_movement_consistencies = []

        return True
