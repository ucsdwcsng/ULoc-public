import numpy as np
from numba import jit

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from config import theta_vals, phi_vals, ant_pos, lamb


def fft_profile_thetaphi(h):
    """Performs 2D FFT for Power Profile over 8 Antennas
    :param h: Antenna CIR stacked matrix
    :return DP: Array for profile at theta_vals
    """

    assert len(h) == 8
    assert len(ant_pos) == 8
    # DP = np.zeros([len(theta_vals), len(phi_vals)], dtype='complex')
    # fft_compute(DP, h, ant_pos, lamb)
    # return DP
    return fft_compute_vectorized(h, ant_pos, lamb)


@jit(nopython=True)
def fft_compute(DP, h, ant_pos, lamb):
    for i in range(len(theta_vals)):
        for j in range(len(phi_vals)):
            # Setups
            # NOTE: Front = Antenna on anchor and tag are in line of sight, anchor antenna directly face tag
            #       Back = Antenna on anchor and tag are NOT in line of sight, signal from tag goes through anchor PCB
            #       X = Origin
            #
            #  NoRotate Front      NoRotate Back      Rotate Front        Rotate Back          Rotate Front
            # ----------------   ----------------   ----------------   ----------------      ----------------
            # | X 3  2  1  0 |   | 0  1  2  3 X |   | 7  6  5  4 X |   | X 4  5  6  7 |      | 7  6  5  4  3 |
            # | 4            |   |            4 |   |            3 |   | 3            |      |             2 |
            # | 5            |   |            5 |   |            2 |   | 2            |      |             1 |
            # | 6            |   |            6 |   |            1 |   | 1            |      |             0 |
            # | 7            |   |            7 |   |            0 |   | 0            |      |               |
            # ----------------   ----------------   ----------------   ----------------      ----------------

            # NoRotate Front
            # horz = np.sum(h[0:4] * np.exp(1j * 2 * np.pi * ant_pos[0:4] * np.sin(theta_vals[i]) * np.cos(phi_vals[j]) / lamb))
            # vert = np.sum(h[4:8] * np.exp(1j * 2 * np.pi * ant_pos[4:8] * np.sin(phi_vals[j]) / lamb))

            # NoRotate Back
            # horz = np.sum(h[0:4] * np.exp(-1j * 2 * np.pi * ant_pos[0:4] * np.sin(theta_vals[i]) * np.cos(phi_vals[j]) / lamb))
            # vert = np.sum(h[4:8] * np.exp(1j * 2 * np.pi * ant_pos[4:8] * np.sin(phi_vals[j]) / lamb))

            # Rotate Front
            horz = np.sum(
                h[4:8] * np.exp(-1j * 2 * np.pi * ant_pos[4:8] * np.sin(theta_vals[i]) * np.cos(phi_vals[j]) / lamb))
            vert = np.sum(h[0:4] * np.exp(1j * 2 * np.pi * ant_pos[0:4] * np.sin(phi_vals[j]) / lamb))

            # Rotate Back
            # horz = np.sum(h[4:8] * np.exp(1j * 2 * np.pi * ant_pos[4:8] * np.sin(theta_vals[i]) * np.cos(phi_vals[j]) / lamb))
            # vert = np.sum(h[0:4] * np.exp(1j * 2 * np.pi * ant_pos[0:4] * np.sin(phi_vals[j]) / lamb))

            DP[i, j] = horz + vert


def fft_compute_vectorized(h, ant_pos, lamb):
    # Rotate Front Configuration (see fft_compute's comments)
    # horz = exp(-j * 2pi * sin(theta) * cos(phi))
    horz_exp = np.tile((np.sin(theta_vals[:, None]) @ np.cos(phi_vals[None]) / lamb)[..., None], (1, 1, 4))
    horz_mat = np.exp(
        -1j * 2 * np.pi * np.tile(ant_pos[4:8][None, None], (len(theta_vals), len(phi_vals), 1)) * horz_exp)

    # vert = exp(j * 2pi * sin(phi))
    vert_exp = np.tile((np.sin(phi_vals[None]) / lamb)[..., None], (len(theta_vals), 1, 4))
    vert_mat = np.exp(
        1j * 2 * np.pi * np.tile(ant_pos[0:4][None, None], (len(theta_vals), len(phi_vals), 1)) * vert_exp)

    FFT = np.dstack((vert_mat, horz_mat))
    DP = FFT @ h[:, None][..., 0]

    return DP
