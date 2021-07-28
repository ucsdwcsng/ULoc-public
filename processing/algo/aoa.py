import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
from config import theta_vals, phi_vals


def abs_normalize(arr):
    if isinstance(arr, np.ndarray):
        assert len(arr.shape) == 1
    else:
        assert isinstance(arr, list)
        assert isinstance(arr[0], complex)
    return np.abs(arr) / np.max(np.abs(arr))


def find_theta_phi(DP):
    assert DP.shape == (180, 180)
    maxti, maxpi = np.unravel_index(np.argmax(DP), DP.shape)
    maxt = theta_vals[maxti] * 180 / np.pi
    maxp = phi_vals[maxpi] * 180 / np.pi
    return maxt, maxp, 1