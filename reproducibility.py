"""Explicit run seeding, preserving the entry point's default seed100 setup."""


def validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise ValueError('RANDOM_SEED / --seed must be an integer in [0, 2**32 - 1].')
    return seed


def initialize_random_state(seed):
    """Seed before constructing datasets/models; return general/train generators.

    Separate generators keep training shuffle independent of validation/test
    iteration. These controls do not promise bitwise equality across hardware.
    """
    import random
    import numpy as np
    import torch

    seed = validate_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    general_generator = torch.Generator().manual_seed(seed)
    train_generator = torch.Generator().manual_seed(seed)
    return general_generator, train_generator


def seed_worker(worker_id):
    """Use each DataLoader worker's torch seed for NumPy/Python as before."""
    import random
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
