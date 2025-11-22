# brainfield_operator/utils/seed.py

import numpy as np
import torch
import random


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[seed] Random seed set to {seed}")
