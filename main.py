import os
import sys

from sample_factory.algorithms.utils.arguments import arg_parser, parse_args
from sample_factory.run_algorithm import run_algorithm

# Needs to be imported to register models and envs
import models
import env


def parse_all_args(argv=None, evaluation=False):
    parser = arg_parser(argv, evaluation=evaluation)
    cfg = parse_args(argv=argv, evaluation=evaluation, parser=parser)
    return cfg


def main():
    """Script entry point."""
    cfg = parse_all_args()
    if "WANDB_API_KEY" in os.environ:
        import wandb
        wandb.init(
            project="sample-factory-nle",
            config=cfg,
            save_code=True,
            name=cfg.experiment,
            sync_tensorboard=True
        )
    status = run_algorithm(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
