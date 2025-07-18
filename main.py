import argparse
import time

from stable_baselines3.common.env_checker import check_env

from config.maze_search.curriculum_config import create_configurable_curriculum
from envs.pybullet.maze_search import MazeEnv, logger
from evaluation.evaluator import evaluate_sb3, record_model_video
from training.curriculum import train_with_curriculum
from training.trainer import train_single_phase


def test_environment(render=True, episodes=10, verbose=False):
    """测试环境功能"""
    logger.info("开始测试迷宫环境...")

    # 创建环境
    render_mode = "human" if render else None
    env = MazeEnv(maze_size=(7, 7), render_mode=render_mode, verbose=verbose)

    # 使用SB3的环境检查器验证环境
    try:
        check_env(env)
        logger.info("环境验证通过！")
    except Exception as e:
        logger.error(f"环境验证失败: {e}")
        return

    # 运行几个回合，使用随机动作
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        logger.info(f"开始回合 {episode + 1}/{episodes}")

        while not done:
            # 随机动作
            action = env.action_space.sample()

            # 执行动作
            obs, reward, done, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            # 打印信息
            if verbose and steps % 20 == 0:
                logger.info(f"步骤: {steps}, 奖励: {reward:.2f}, 累计奖励: {total_reward:.2f}")
                if 'distance_to_ball' in info:
                    logger.info(f"到球距离: {info['distance_to_ball']:.2f}")

            # 渲染
            if render:
                env.render()
                time.sleep(0.01)  # 降低速度以便观察

            # 如果回合太长，提前结束
            if steps >= 500:
                logger.info("回合过长，提前结束")
                break

        logger.info(f"回合 {episode + 1} 结束，总步数: {steps}, 总奖励: {total_reward:.2f}")

    # 关闭环境
    env.close()
    logger.info("环境测试完成！")


def train_curriculum(render=False, resume=False, phase=None, verbose=False):
    """使用课程学习训练模型"""
    logger.info("开始使用课程学习训练迷宫环境...")

    # 创建课程学习配置
    curriculum_config = create_configurable_curriculum()

    # 训练
    results = train_with_curriculum(MazeEnv, curriculum_config, resume=resume, phase_set=phase, verbose=verbose)

    # 打印结果
    for phase, complete in results.items():
        logger.info(f"{phase}: {'完成' if complete else '未完成'}")

    logger.info("课程学习训练完成！")

    return results


def train_sb3(resume=False, verbose=False):
    """使用课程学习训练模型"""
    logger.info("开始使用课程学习训练迷宫环境...")

    # 创建课程学习配置
    curriculum_config = create_configurable_curriculum()

    # 训练
    results = train_single_phase(
        env_class=MazeEnv,
        config=curriculum_config,
        algorithm="SAC",
        phase=1,
        verbose=verbose,
        resume=resume
    )

    # 打印结果
    for phase, complete in results.items():
        logger.info(f"{phase}: {'完成' if complete else '未完成'}")

    logger.info("课程学习训练完成！")

    return results


def evaluate_model(model_path, episodes=5, phase=3, render=True, verbose=False):
    """评估训练好的模型"""
    logger.info(f"开始评估模型: {model_path}")
    import os
    # model_path = os.path.join(PROJECT_ROOT,
    #                           'checkpoints',
    #                           'maze_search',
    #                           'sb3_models',
    #                           f'SAC_phase{phase}',
    #                           )
    # 评估模型
    # 创建课程学习配置
    curriculum_config = create_configurable_curriculum()
    mean_reward, std_reward = evaluate_sb3(
        model_path=model_path,
        env_class=MazeEnv,
        curriculum_config=curriculum_config,
        episodes=episodes,
        phase=phase,
        render_mode="human" if render else None,
        verbose=verbose,
    )

    if mean_reward is not None:
        logger.info(f"评估结果 - 平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    else:
        logger.error("评估失败")

    return mean_reward, std_reward


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="机器人迷宫寻路强化学习项目")

    # 添加子命令
    subparsers = parser.add_subparsers(dest="command", help="命令")

    # 测试环境命令
    test_parser = subparsers.add_parser("test", help="测试环境")
    test_parser.add_argument("--episodes", type=int, default=3, help="测试回合数")
    test_parser.add_argument("--no-render", action="store_true", help="禁用渲染")
    test_parser.add_argument("--verbose", action="store_true", help="显示详细信息")

    # 训练命令
    train_parser = subparsers.add_parser("train_sb3", help="训练模型")
    train_parser.add_argument("--resume", action="store_true", help="从已保存的模型继续训练")
    train_parser.add_argument("--verbose", action="store_true", help="显示详细信息")

    # 课程学习命令
    curriculum_parser = subparsers.add_parser("curriculum", help="使用课程学习训练")
    curriculum_parser.add_argument("--render", action="store_true", help="启用渲染")
    curriculum_parser.add_argument("--resume", action="store_true", help="从已保存的模型继续训练")
    curriculum_parser.add_argument("--phase", type=int, default=None, help="加载阶段模型")
    curriculum_parser.add_argument("--verbose", action="store_true", help="显示详细信息")

    # 评估命令
    eval_parser = subparsers.add_parser("eval", help="评估模型")
    eval_parser.add_argument("--model", type=str, required=False, help="模型路径")
    eval_parser.add_argument("--phase", type=int, default=3, help="加载阶段模型")
    eval_parser.add_argument("--episodes", type=int, default=5, help="评估回合数")
    eval_parser.add_argument("--no-render", action="store_true", help="禁用渲染")
    eval_parser.add_argument("--verbose", action="store_true", help="显示详细信息")

    # 新增: 视频录制命令
    record_parser = subparsers.add_parser("record", help="录制模型视频")
    record_parser.add_argument("--model", type=str, required=True, help="模型路径")
    record_parser.add_argument("--phase", type=int, default=3, help="加载阶段模型")
    record_parser.add_argument("--episodes", type=int, default=1, help="录制回合数")
    record_parser.add_argument("--video-length", type=int, default=1000, help="每个视频的最大帧数")
    record_parser.add_argument("--video-folder", type=str, default="videos", help="视频保存文件夹")
    record_parser.add_argument("--verbose", action="store_true", help="显示详细信息")

    # 解析命令行参数
    args = parser.parse_args()

    # 执行相应命令
    if args.command == "test":
        test_environment(render=not args.no_render, episodes=args.episodes, verbose=args.verbose)
    elif args.command == "train_sb3":
        train_sb3(resume=args.resume, verbose=args.verbose)
    elif args.command == "curriculum":
        train_curriculum(render=args.render, resume=args.resume, phase=args.phase, verbose=args.verbose)
    elif args.command == "eval":
        evaluate_model(model_path=args.model,
                       phase=args.phase,
                       episodes=args.episodes,
                       render=not args.no_render,
                       verbose=args.verbose)
    elif args.command == "record":
        # 新增: 录制视频的函数
        record_model_video(
            model_path=args.model,
            env_class=MazeEnv,
            curriculum_config=create_configurable_curriculum(),
            episodes=args.episodes,
            phase=args.phase,
            video_length=args.video_length,
            video_folder=args.video_folder,
            verbose=args.verbose
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
