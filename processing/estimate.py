import os
import numpy as np
import matplotlib.pyplot as plt
from p_tqdm import p_map

from load import parse
from algo import cir, fft, aoa


def load_cir(exp_data, zero_data, tag_addr, upsample):
    # Load calibration data ("zero-AoA")
    ap_zero = parse.load_log(zero_data, interp_cir=upsample)
    ap_zero = parse.reformat_ap_data(ap_data=ap_zero, interp_cir=upsample, select_tag_addr=tag_addr)
    ap_zero = cir.extract_fp(ap_zero)
    zero_deg_offset = cir.calc_0deg_offset(ap_zero['cir_fp'])
    zero_dag_offset_sync = cir.calc_adj_0deg_offset(ap_zero['cir_sync'])

    # Load, and correct actual data against calibration data
    ap_data = parse.load_log(exp_data, interp_cir=upsample)
    ap_data = parse.reformat_ap_data(ap_data=ap_data, interp_cir=upsample, select_tag_addr=tag_addr)
    ap_data = cir.extract_fp(ap_data)
    ap_data['cir_fp'] = cir.correct_cir(ap_data['cir_fp'], zero_deg_offset)
    ap_data['cir_sync'] = cir.correct_adj_cir(ap_data['cir_sync'], zero_dag_offset_sync)
    return ap_data


def fft_worker(h):
    assert isinstance(h, np.ndarray)
    assert h.ndim == 1 and h.shape == (8,)

    h = h / np.max(np.abs(h))
    profile = np.abs(fft.fft_profile_thetaphi(h))
    # Normalize
    profile /= np.max(profile)
    theta_est, phi_est, _ = aoa.find_theta_phi(profile)
    return theta_est, phi_est, profile


def estimate_aoa(ap_data, start_idx=0):
    output = p_map(fft_worker, ap_data['cir_fp'][start_idx:])  # List of each packet's (theta_est, phi_est, profile)
    theta_est = np.array([x[0] for x in output])
    phi_est = np.array([x[1] for x in output])
    profiles = np.array([x[2] for x in output])
    return theta_est, phi_est, profiles


if __name__ == '__main__':
    data_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

    exp_data_file = os.path.join(data_root, 'example_data', 'uloc_tracking_1_20210427_182741_207A357A4653.log')
    exp_calibration_file = os.path.join(data_root, 'example_data', 'uloc_zero_0_20210427_181016_207A357A4653.log')

    cir_data = load_cir(exp_data=exp_data_file, zero_data=exp_calibration_file, tag_addr='0000', upsample=8)
    aoa_data = estimate_aoa(cir_data)

    # Plotting
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(aoa_data[0], '.')
    axes[1].plot(aoa_data[1], '.')

    axes[0].set_title('Theta Estimated over Time')
    axes[0].set_ylabel('Theta (deg)')
    axes[1].set_title('Phi Estimated over Time')
    axes[1].set_ylabel('Phi (deg)')

    for ax in axes:
        ax.set_xlabel('Packet Index')
        ax.set_ylim(-90, 90)

    plt.tight_layout()
    plt.show()
