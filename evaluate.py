import os
import sys
import time
from collections import deque

import numpy as np
import torch

from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.algorithms.utils.action_distributions import ContinuousActionDistribution
from sample_factory.algorithms.utils.algo_utils import ExperimentStatus
from sample_factory.algorithms.utils.arguments import arg_parser, parse_args, load_from_checkpoint
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log, AttrDict

# Needs to be imported to register models and envs
import models as my_models
import env as my_envs


TARGET_NUM_EPISODES = 512

def enjoy(cfg, max_num_frames=1e9, target_num_episodes=TARGET_NUM_EPISODES):
    """
    This is a modified version of original appo.enjoy_appo.enjoy function,
    modified to have an episode limit.
    """
    cfg = load_from_checkpoint(cfg)

    cfg.env_frameskip = 1
    cfg.num_envs = 1
    cfg.use_aicrowd_gym = True

    def make_env_func(env_config):
        return create_env(cfg.env, cfg=cfg, env_config=env_config)

    env = make_env_func(AttrDict({'worker_index': 0, 'vector_index': 0}))

    # sample-factory defaults to work with multiagent environments,
    # but we can wrap a single-agent env into one of these like this
    env = MultiAgentWrapper(env)

    log.info("Num agents {}".format(env.num_agents))

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)

    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    checkpoints = LearnerWorker.get_checkpoints(LearnerWorker.checkpoint_dir(cfg, policy_id))
    checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict['model'])

    episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    true_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    num_frames = 0
    num_episodes = 0

    obs = env.reset()
    rnn_states = torch.zeros([env.num_agents, get_hidden_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = np.zeros(env.num_agents)
    finished_episode = [False] * env.num_agents

    with torch.no_grad():
        while num_frames < max_num_frames and num_episodes < target_num_episodes:
            obs_torch = AttrDict(transform_dict_observations(obs))
            for key, x in obs_torch.items():
                obs_torch[key] = torch.from_numpy(x).to(device).float()

            policy_outputs = actor_critic(obs_torch, rnn_states, with_action_distribution=True)

            # sample actions from the distribution by default
            actions = policy_outputs.actions

            actions = actions.cpu().numpy()

            rnn_states = policy_outputs.rnn_states

            obs, rew, done, infos = env.step(actions)

            episode_reward += rew
            num_frames += 1

            for agent_i, done_flag in enumerate(done):
                if done_flag:
                    finished_episode[agent_i] = True
                    episode_rewards[agent_i].append(episode_reward[agent_i])
                    true_rewards[agent_i].append(infos[agent_i].get('true_reward', episode_reward[agent_i]))
                    log.info('Episode finished for agent %d at %d frames. Reward: %.3f, true_reward: %.3f', agent_i, num_frames, episode_reward[agent_i], true_rewards[agent_i][-1])
                    rnn_states[agent_i] = torch.zeros([get_hidden_size(cfg)], dtype=torch.float32, device=device)
                    episode_reward[agent_i] = 0
                    num_episodes += 1

            if all(finished_episode):
                finished_episode = [False] * env.num_agents
                avg_episode_rewards_str, avg_true_reward_str = '', ''
                for agent_i in range(env.num_agents):
                    avg_rew = np.mean(episode_rewards[agent_i])
                    avg_true_rew = np.mean(true_rewards[agent_i])
                    if not np.isnan(avg_rew):
                        if avg_episode_rewards_str:
                            avg_episode_rewards_str += ', '
                        avg_episode_rewards_str += f'#{agent_i}: {avg_rew:.3f}'
                    if not np.isnan(avg_true_rew):
                        if avg_true_reward_str:
                            avg_true_reward_str += ', '
                        avg_true_reward_str += f'#{agent_i}: {avg_true_rew:.3f}'

                log.info('Avg episode rewards: %s, true rewards: %s', avg_episode_rewards_str, avg_true_reward_str)
                log.info('Avg episode reward: %.3f, avg true_reward: %.3f', np.mean([np.mean(episode_rewards[i]) for i in range(env.num_agents)]), np.mean([np.mean(true_rewards[i]) for i in range(env.num_agents)]))

    env.close()

    return ExperimentStatus.SUCCESS, np.mean(episode_rewards)


def parse_all_args(argv=None, evaluation=True):
    parser = arg_parser(argv, evaluation=evaluation)
    cfg = parse_args(argv=argv, evaluation=evaluation, parser=parser)
    return cfg


def main():
    """Evaluation entry point."""
    cfg = parse_all_args()
    _ = enjoy(cfg)
    return


if __name__ == '__main__':
    sys.exit(main())
