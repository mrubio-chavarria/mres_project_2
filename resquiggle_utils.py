#!/home/mario/anaconda3/envs/project2_venv/bin python

"""
DESCRIPTION:
The functions in this file solve the problem of reading from the 
resquiggled fast5 files.
"""

# Libraries
import h5py
from tombo import tombo_helper, tombo_stats, resquiggle
import mappy
import torch
import numpy as np


# Functions
def collapse(window):
    """
    DESCRIPTION:
    This function:
    1. Divides the sequence by segment fragment.
    2. Collapses the repeated letters within a segment.
    :param window: [dict] the instance to predict with sequence, the 
    target value and signal, the ONT values.
    :return: [dict] the same window but the sequence has just one 
    letter per segment (not per signal value).
    """
    window['fragments'] = [len(segment) for segment in window['sequence'].split('*') if segment]
    window['sequence'] = ''.join([
        ''.join(set(segment))
        for segment in window['sequence'].split('*')
    ])
    return window


def parse_resquiggle(read_file, reference_file, bandwidth=6000, read=False, norm='usual'):
    """
    DESCRIPTION:
    A function to read the information in a FAST5 file in which the 
    resquiggle has been performed. The function applies the normalisation
    on the read data and trims all the non-relevant regions.
    IMPORTANT: when computing, you are returning the normalised signal,
    when reading, the raw signal.
    :param read_file: [str] route to the fast5 file with the resquiggled
    sequence.
    :param reference_file: [str] route to the fasta file with the reference.
    :param bandwith: [int] bandwith limit to compute the event alignment 
    during DTW. 
    :param read: [bool] a flag to indicate if the resquiggling will be read
    from the file or recomputed.
    :return: [tuple] there are three outputs. The first is a np.ndarray with
    the positions denoting the segments in the signal. The second is the str
    with the trimmed sequence after normalise. The third is a 
    np.ndarray with the trimmed and normalised signal of the fast5 file.
    """
    fast5_data = h5py.File(read_file, 'r')
    # Print read file for debug
    # Read the resquiggling from file or compute it
    if read:
        # Obtain the events
        events = fast5_data['Analyses']['RawGenomeCorrected_000']['BaseCalled_template']['Events']
        # Get raw signal
        read_name = list(fast5_data['Raw']['Reads'].keys())[0]  # They will always be single read files
        signal = fast5_data['Raw']['Reads'][read_name]['Signal'].value.astype('float64')
        read_start_rel_to_raw = events.attrs['read_start_rel_to_raw']
        # Format the events
        last_event = events.value.tolist()[-1]
        events = map(lambda x: (x[2], x[4].decode('utf-8')), events.value.tolist())
        # Create sequence and segs
        sequence = []
        segs = []
        [(sequence.append(event[-1]), segs.append(event[0])) for event in events]
        segs.append(last_event[-3] + last_event[-2])
        segs = np.array(segs)
        sequence = ''.join(sequence)
        # Trim the signal
        norm_signal = signal[read_start_rel_to_raw:read_start_rel_to_raw + segs[-1]]
        # Normalise signal
        if norm == 'usual':
            mu = np.mean(signal)
            sd = np.std(signal)
            norm_signal = (norm_signal - mu) / sd
    else:
        # Set parameters for resquiggling
        aligner = mappy.Aligner(reference_file, preset=str('map-ont'), best_n=1)
        seq_samp_type = tombo_helper.get_seq_sample_type(fast5_data)
        std_ref = tombo_stats.TomboModel(seq_samp_type=seq_samp_type)
        rsqgl_params = tombo_stats.load_resquiggle_parameters(seq_samp_type)
        rsqgl_params = rsqgl_params._replace(bandwidth=bandwidth) 
        # Extract data from FAST5
        map_results = resquiggle.map_read(fast5_data, aligner, std_ref)
        all_raw_signal = tombo_helper.get_raw_read_slot(fast5_data)['Signal'][:]
        if seq_samp_type.rev_sig:
            all_raw_signal = all_raw_signal[::-1]
        map_results = map_results._replace(raw_signal=all_raw_signal)
        # Detect events in raw signal
        num_events = tombo_stats.compute_num_events(all_raw_signal.shape[0],
            len(map_results.genome_seq), rsqgl_params.mean_obs_per_event)
        # Segmentation
        valid_cpts, norm_signal, scale_values = resquiggle.segment_signal(
            map_results, num_events, rsqgl_params)
        # Normalisation
        event_means = tombo_stats.compute_base_means(norm_signal, valid_cpts)
        # Alignment
        dp_results = resquiggle.find_adaptive_base_assignment(valid_cpts,
            event_means, rsqgl_params, std_ref, map_results.genome_seq)
        # Trim normalised signal
        norm_signal = norm_signal[
            dp_results.read_start_rel_to_raw:
            dp_results.read_start_rel_to_raw + dp_results.segs[-1]]
        # Segments specifiying in order the number of signal per base in the
        # sequence, there are the same of segments and normalised 
        # sequence
        segs = resquiggle.resolve_skipped_bases_with_raw(
            dp_results, norm_signal, rsqgl_params)
        sequence = dp_results.genome_seq
    return segs, sequence, norm_signal


def window_resquiggle(segs, genome_seq, norm_signal, window_size=300, overlap=0.9):
    """
    DESCRIPTION:
    A function to window the normalised signal obtained from the function 
    parse_resquiggle. The ndows are returned in the form of a dict to be 
    consistent with downwards pytorch integration.
    :param segs: [list] positions denoting the segments in norm_signal.
    :param genome_seq: [str] trimmed sequence after normalise.
    :param norm_signal: [torch.FloatTensor] trimmed and normalised signal of the fast5
    file.
    :return: [list] pairs of sequence and signal with 1-to-1 correspondence.
    """
    # Relate signal to sequence
    # $: symbol to fill
    # *: symbol to distinguish between segments
    initial_seq_signal = [((genome_seq[i] + '$') * (segs[i+1] - segs[i]))[:-1] 
        for i in range(len(segs[:-1]))]
    seq_signal = '*'.join(initial_seq_signal) + '$'
    # Perform the windowing over the signal
    windows = []
    for i in range(0, len(norm_signal), int(round((1 - overlap) * window_size))):
        window = {'signal_indeces': (i, i+window_size),
                'sequence': seq_signal[2*i:2*(i+window_size)].replace('$', ''),
                'signal': torch.FloatTensor(norm_signal[i:i+window_size])}
        windows.append(window)
    # Collapse the repeated bases for all the windows
    windows = [collapse(window) for window in windows]
    # Filter uncompleted windows and return
    windows = windows[:-int(round(1/(1-overlap)))]
    return windows