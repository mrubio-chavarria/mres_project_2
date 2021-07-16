#!/home/mario/anaconda3/envs/project2_venv/bin python

"""
DESCRIPTION:
The functions in this file solve the problem of reading from the 
resquiggled fast5 files.
"""

# Libraries
import h5py
from ont_fast5_api.fast5_interface import get_fast5_file
from tombo import tombo_helper, tombo_stats, resquiggle
import mappy
import torch


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


def parse_resquiggle(read_file, reference_file):
    """
    DESCRIPTION:
    A function to read the information in a FAST5 file in which the 
    resquiggle has been performed. The function applies the normalisation
    on the read data and trims all the non-relevant regions.
    :param read_file: [str] route to the fast5 file with the resquiggled
    sequence.
    :param reference_file: [str] route to the fasta file with the reference.
    :return: [tuple] there are three outputs. The first is a np.ndarray with
    the positions denoting the segments in the signal. The second is the str
    with the trimmed sequence after normalise. The third is a 
    np.ndarray with the trimmed and normalised signal of the fast5 file.
    """
    fast5_data = h5py.File(read_file, 'r')
    # Set parameters for resquiggling
    aligner = mappy.Aligner(reference_file, preset=str('map-ont'), best_n=1)
    seq_samp_type = tombo_helper.get_seq_sample_type(fast5_data)
    std_ref = tombo_stats.TomboModel(seq_samp_type=seq_samp_type)
    rsqgl_params = tombo_stats.load_resquiggle_parameters(seq_samp_type)
    rsqgl_params = rsqgl_params._replace(bandwidth=6000) 
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
    # sequence, there are roughly the same of segments and normalised 
    # signals
    segs = resquiggle.resolve_skipped_bases_with_raw(
        dp_results, norm_signal, rsqgl_params)
    return segs, dp_results.genome_seq, norm_signal


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
    seq_signal = '*'.join([((genome_seq[i] + '$') * len(norm_signal[segs[i]:segs[i+1]]))[:-1] 
        for i in range(len(segs[:-1]))]) + '$'
    # Perform the windowing over the signal
    windows = [{
        'signal_indeces': (i, i+window_size),
        'sequence': seq_signal[2*i:2*(i+window_size)].replace('$', ''),
        'signal': torch.FloatTensor(norm_signal[i:i+window_size])} 
        for i in range(0, len(norm_signal), int(round((1 - overlap) * window_size)))]
    # Collapse the repeated bases for all the windows
    windows = [collapse(window) for window in windows]
    # Filter uncompleted windows and return
    return windows[:-int(round(1/(1-overlap)))]