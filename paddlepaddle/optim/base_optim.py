import argparse


class BaseOptim(object):
    """Base class for optimizer"""

    def __init__(self, opts) -> None:
        self.epsilon = 1e-8
        self.learning_rate = getattr(opts, "scheduler.lr", 0.1)
        self.weight_decay = getattr(opts, "optim.weight_decay", 4e-5)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser
