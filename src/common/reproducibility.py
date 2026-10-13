"""Reproducibility utilities — seed management for numpy, random, torch."""
from __future__ import annotations

import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
