import os
import gym
import numpy as np

import nle
from PIL import Image, ImageFont, ImageDraw

from obs_wrappers import RenderCharImagesWithNumpyWrapper, VectorFeaturesWrapper
from sample_factory.envs.env_registry import global_env_registry


class RootNLEWrapper(gym.Wrapper):
    """Some root-level additions to the NLE environment"""
    def __init__(self, env):
        super().__init__(env)

    def seed(self, *args):
        # Nethack does not allow seeding, so monkey-patch disable it here
        return


def make_custom_env_func(full_env_name, cfg=None, env_config=None):
    env = RootNLEWrapper(gym.make("NetHackChallenge-v0", observation_keys=["tty_chars", "tty_colors", "blstats", "message"]))
    if full_env_name == "nle_competition_image_obs":
        env = VectorFeaturesWrapper(
            RenderCharImagesWithNumpyWrapper(env, font_size=9, crop_size=12, rescale_font_size=(6, 6))
        )
    else:
        raise ValueError(f"Env does not exist {full_env_name}")
    return env


global_env_registry().register_env(
    env_name_prefix='nle_',
    make_env_func=make_custom_env_func,
)
