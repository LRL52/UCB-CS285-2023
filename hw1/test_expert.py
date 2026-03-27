"""
测试专家策略的性能
"""
import gym
import numpy as np
from cs285.policies.loaded_gaussian_policy import LoadedGaussianPolicy
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger

def test_expert_policy(env_name, expert_policy_file, num_rollouts=5, save_video=False):
    """
    测试专家策略并打印统计信息

    Args:
        env_name: 环境名称
        expert_policy_file: 专家策略文件路径
        num_rollouts: 测试轨迹数量
        save_video: 是否保存视频
    """
    # 初始化
    ptu.init_gpu(use_gpu=False, gpu_id=0)

    # 创建环境
    env = gym.make(env_name, render_mode=None)
    max_ep_len = env.spec.max_episode_steps

    # 获取 fps
    if hasattr(env, 'model'):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata.get('render_fps', 30)

    # 加载专家策略
    print(f'Loading expert policy from {expert_policy_file}...')
    expert_policy = LoadedGaussianPolicy(expert_policy_file)
    expert_policy.to(ptu.device)
    print('Expert policy loaded successfully!\n')

    # 使用框架的 sample_n_trajectories 函数采样
    print(f'Collecting {num_rollouts} rollouts...')
    paths = utils.sample_n_trajectories(env, expert_policy, num_rollouts, max_ep_len, render=save_video)

    # 计算统计信息
    returns = [path["reward"].sum() for path in paths]
    episode_lengths = [len(path["reward"]) for path in paths]

    # 打印每条轨迹的结果
    for i, ret in enumerate(returns):
        print(f'Rollout {i+1}/{num_rollouts}: Return: {ret:.2f}, Length: {episode_lengths[i]}')

    # 保存视频
    if save_video:
        import os
        os.makedirs('videos', exist_ok=True)
        logdir = f'videos/{env_name}_expert'
        logger = Logger(logdir)
        logger.log_paths_as_videos(
            paths, 0,
            fps=fps,
            max_videos_to_save=num_rollouts,
            video_title='expert_rollouts'
        )
        print(f'\nVideos saved to {logdir}/')

    # 打印统计信息
    print('\n' + '='*50)
    print(f'Environment: {env_name}')
    print(f'Number of rollouts: {num_rollouts}')
    print('-'*50)
    print(f'Average Return: {np.mean(returns):.2f} ± {np.std(returns):.2f}')
    print(f'Max Return: {np.max(returns):.2f}')
    print(f'Min Return: {np.min(returns):.2f}')
    print(f'Average Episode Length: {np.mean(episode_lengths):.1f}')
    print('='*50)

    return returns, episode_lengths


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Ant-v4',
                        choices=['Ant-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Hopper-v4'])
    parser.add_argument('--num_rollouts', type=int, default=5)
    parser.add_argument('--save_video', action='store_true', help='Save video of rollouts')
    args = parser.parse_args()

    # 专家策略文件路径
    expert_policy_file = f'cs285/policies/experts/{args.env_name.split("-")[0]}.pkl'

    # 测试专家策略
    test_expert_policy(args.env_name, expert_policy_file, args.num_rollouts, save_video=args.save_video)
