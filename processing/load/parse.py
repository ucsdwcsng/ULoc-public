import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import re
import json
import math
import traceback
import numpy as np
from config import CIR_LEN
from algo.cir import interpolate_cir

# Parse (and interpolate) CIR
_parse_hex_complex_matcher = re.compile('([0-9a-f][0-9a-f][0-9a-f][0-9a-f])([0-9a-f][0-9a-f][0-9a-f][0-9a-f])')


def _s16(s):
    value = int(s, 16)
    return -(value & 0x8000) | (value & 0x7fff)


def _parse_hex_complex(s):
    # Log is already in correct endianness
    m = _parse_hex_complex_matcher.search(s)
    if m is None:
        return float('nan')
    else:
        x = _s16(m.group(1)) + 1.j * _s16(m.group(2))
        # print('{} -> {} + {}j -> {}'.format(s, m.group(1), m.group(2), x))
        return x


def _parse_cir(s, interp_cir):
    cir = [_parse_hex_complex(x) for x in s.split()]  # Split to each tap
    assert len(cir) == CIR_LEN, f'CIR has length {len(cir)}, expected {CIR_LEN}!! Check config.py and make sure you' \
                                f'set the CIR_LEN to the expected length!'
    if interp_cir != 1:
        cir = interpolate_cir(cir, interp_cir)
        assert len(cir) > 1
    return cir


# Parse Data Packet
_packet_info_parser = re.compile(
    'C([\d])R([\da-f][\da-f])T([\da-f][\da-f][\da-f][\da-f]) ((?:[\dda-f]+ )+)S(\d+)F(\d+) \d+Hz')


def _process_ap_one_packet(f, n_chips=8, suppress=True, interp_cir=1):
    packet_data = dict()
    line = ''
    # Find START COLLECT
    while line.find('START COLLECT') == -1:
        line = f.readline()
        if not line:
            return None

    # Process all chips
    while line.find('COLLECT DONE') == -1:
        line = f.readline()
        if not line:  # EOF
            if not suppress:
                print("Log Ended")
            break
        m = _packet_info_parser.search(line)

        if m is not None:
            try:  # NOTE: Parse functions takes care of reporting errors
                antenna_id = int(m.group(1), 10)
                packet_id = int(m.group(2), 16)
                tag_addr = m.group(3)
                cir = _parse_cir(m.group(4), interp_cir=interp_cir)
                sfd = int(m.group(5), 10)
                fpi = int(m.group(6), 10)

                packet_data[antenna_id] = (cir, sfd / 128.0 * 2.0 * math.pi, fpi / 64.0, packet_id, tag_addr)
            except:
                if not suppress:
                    traceback.print_exc()

    # Accounting...
    if len(packet_data.keys()) is not n_chips:
        if not suppress:
            print(
                '\n[Log Parser] Warning: Dropped packet with data from only {}/{} chips'.format(len(packet_data.keys()), n_chips))
        return None

    packet_ids_antenna = np.array([packet_data[i][4] for i in range(n_chips)])
    if not np.all(packet_ids_antenna == packet_ids_antenna[0]):
        print('\n[Log Parser] Warning: Dropped packet with inconsistent packet IDs across antennas:\n' + str(
            packet_ids_antenna))
        return None
    else:
        print('.', end='')
        return packet_data


class PacketIDUnwrapper:
    """ Unwraps the UWB packet ID which rolls over at 256 """

    def __init__(self, rollover_at=256):
        self.last_id = -1
        self.unwrap_cnt = 0
        self.rollover = rollover_at

    def unwrap(self, current_id):
        assert isinstance(current_id, int)
        assert current_id < self.rollover, '[Packet ID Unwrapper] Current ID should never ≥ number that rolls over'
        if current_id < self.last_id:
            self.unwrap_cnt += 1
        self.last_id = current_id
        return current_id + self.unwrap_cnt * self.rollover


def load_log(file_path, suppress=False, interp_cir=1):
    """
    Loads, parse and upsample ULoc AP CIR Data
    Args:
        file_path: Path to AP Log
        suppress: Whether to suppress
        interp_cir: How many times to upsample

    Returns:

    """
    f = open(file_path, 'r')
    header = f.readline()
    exp_type = 'unknown'
    try:
        info = json.loads(header)
        exp_type = info['type']
    except:
        if not suppress:
            raise Exception(f'Error loading header from {file_path} -- invalid data?')

    f.seek(0, 2)
    file_size = f.tell()
    f.seek(0)

    if exp_type == 'anchor':
        ap_data_all = list()
        while True:
            if f.tell() == file_size:
                break
            ap_data_one_packet = _process_ap_one_packet(f, suppress=suppress, interp_cir=interp_cir)
            if ap_data_one_packet is None:
                continue
            ap_data_all.append(ap_data_one_packet)
        assert len(ap_data_all) != 0, 'No valid AP data parsed!'
        return ap_data_all

    else:
        raise NotImplementedError("Unsupported log type")


def reformat_ap_data(ap_data, interp_cir=1, select_tag_addr=None):
    assert isinstance(ap_data, list)

    if select_tag_addr is not None:
        # For multi tag use
        filter_mask = [i[0][4] == select_tag_addr for i in ap_data]
        ap_data = [ap_data[i] for i in np.where(filter_mask)[0]]

    assert len(ap_data) != 0, 'AP Data after tag filtering is empty! Check tag filter?'
    packet_id_unwrapper = PacketIDUnwrapper()
    num_of_pacs = len(ap_data)
    num_of_taps = len(ap_data[0][0][0])

    cir_packets = np.zeros([num_of_pacs, num_of_taps, 8], dtype='complex')
    packet_ids = np.zeros(num_of_pacs, dtype=np.int64)
    fp_idxs = np.zeros([num_of_pacs, 8], dtype=np.float64)
    sfd_data = np.zeros([num_of_pacs, 8], dtype=np.float64)
    valid_pak_idx = np.zeros(num_of_pacs, dtype=bool)

    for packet, nth_packet in zip(ap_data, range(len(ap_data))):
        assert sorted(list(packet.keys())) == list(range(8))
        for chip in range(8):
            # Parsed log format is a list (pak id) of dict (antenna) of tuple
            # defined in parse/_process_ap_one_packet: cir[chipno] = (xx, xx, xx, xx, ...)
            assert len(packet[chip]) == 5, 'Unrecognized format!'
            cir, sfd, fpi, packet_id, tag_addr = packet[chip]
            assert len(cir) == CIR_LEN * interp_cir
            cir_packets[nth_packet, :, chip] = cir
            # FIXME: rangeno bug??
            if chip == 3:  # IMPORTANT: Use Chip 3's packet ID for this packet
                packet_id_true = packet_id_unwrapper.unwrap(packet_id)
                packet_ids[nth_packet] = packet_id_true
                valid_pak_idx[nth_packet] = True

            fp_idxs[nth_packet, chip] = fpi
            sfd_data[nth_packet, chip] = sfd

    return {
        'cir': cir_packets[valid_pak_idx],
        'packet_ids': packet_ids[valid_pak_idx],
        'fp_idxs': fp_idxs[valid_pak_idx],
        'sfd_data': sfd_data[valid_pak_idx],
        'interp_cir': interp_cir
    }


if __name__ == '__main__':
    # For Testing
    load_log('test.log')
