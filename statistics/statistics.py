#!/home/mario/anaconda3/envs/project2_venv/bin python

"""
DESCRIPTION:
An script to retrieve the information generated during
the resquiggling from the fastq files.
"""

import h5py
import os
import pandas as pd
import csv
import numpy as np
from pytictoc import TicToc
from tombo import tombo_helper, tombo_stats, resquiggle
import mappy
from torch import mean


if __name__ == "__main__":

    # Process the 3xr6 dataset
    t = TicToc()
    t.tic()
    dataset_folder = "/home/mario/Projects/project_2/databases/working_3xr6/reads"
    reference_file = dataset_folder + "/" + "reference.fasta"
    reads_data = []
    for flowcell in ['flowcell1', 'flowcell2', 'flowcell3']:
        flowcell_folder = dataset_folder + '/' + flowcell + '/' + 'single'
        for subfolder in os.listdir(flowcell_folder):
            if subfolder.endswith('txt') or subfolder.endswith('index'):
                continue
            subfolder = flowcell_folder + '/'  + subfolder
            for read_file in os.listdir(subfolder):
                read_name = read_file
                if not read_file.endswith('fast5'):
                    continue
                read_file = subfolder + '/' + read_file
                try:
                    fast5_data = h5py.File(read_file, 'r')
                    template = fast5_data['Analyses']['RawGenomeCorrected_000']['BaseCalled_template']
                except:
                    # Parsing: WRONG
                    # Alignment: WRONG
                    clipped_start = -1
                    clipped_end = -1
                    mapped_start = -1
                    mapped_end = -1
                    num_deletions = -1
                    num_insertions = -1
                    num_matches = -1
                    num_events = -1
                    num_mismatches = -1
                    signal_matching_score = -1
                    raw_data = -1  # fast5_data['Raw']['Reads'][list(fast5_data['Raw']['Reads'].keys())[0]]
                    read_id = -1  # raw_data.attrs['read_id'].decode('UTF-8')
                    raw_signal_length = -1  # raw_data['Signal'].value.shape[0]
                    fastq = -1  # fast5_data['Analyses']['Basecall_1D_000']['BaseCalled_template']['Fastq'].value
                    sequence = -1  # fastq.split('\n')[1]
                    at_content = -1  # sequence.count('A') +  sequence.count('T')
                    gc_content = -1  # sequence.count('G') +  sequence.count('C')
                    # at_content, gc_content = at_content / (at_content + gc_content), gc_content / (at_content + gc_content)
                    sequence_length = -1  # len(sequence)
                    aligned_section_length = -1
                    failed_parsing = True
                    failed_alignment = True
                    mean_q_score = -1
                    reads_data.append(
                        (read_id, raw_signal_length, sequence_length, clipped_start, clipped_end, mapped_start, mapped_end, num_deletions, num_insertions, num_matches, num_mismatches, num_events, signal_matching_score, failed_parsing, failed_alignment, aligned_section_length, at_content, gc_content, mean_q_score)
                    )
                    continue
                status = template.attrs['status']
                if status == 'Alignment not produced':
                    # Parsing: OK
                    # Alignment: WRONG
                    raw_data = fast5_data['Raw']['Reads'][list(fast5_data['Raw']['Reads'].keys())[0]]
                    read_id = raw_data.attrs['read_id'].decode('UTF-8')
                    raw_signal_length = raw_data['Signal'].value.shape[0]
                    signal_matching_score = -1
                    fastq = fast5_data['Analyses']['Basecall_1D_000']['BaseCalled_template']['Fastq'].value
                    sequence = fastq.split('\n')[1]
                    at_content = sequence.count('A') +  sequence.count('T')
                    gc_content = sequence.count('G') +  sequence.count('C')
                    at_content, gc_content = at_content / (at_content + gc_content), gc_content / (at_content + gc_content)
                    sequence_length = len(sequence)
                    clipped_start = -1
                    clipped_end = -1
                    mapped_start = -1
                    mapped_end = -1
                    num_deletions = -1
                    num_insertions = -1
                    num_matches = -1
                    num_mismatches = -1
                    num_events = -1
                    aligned_section_length = -1
                    failed_parsing = False
                    failed_alignment = True
                    mean_q_score = -1
                else:
                    try:
                        alignment = template['Alignment']
                    except KeyError:
                        # Parsing: WRONG
                        # Alignment: WRONG
                        clipped_start = -1
                        clipped_end = -1
                        mapped_start = -1
                        mapped_end = -1
                        num_deletions = -1
                        num_insertions = -1
                        num_matches = -1
                        num_events = -1
                        num_mismatches = -1
                        signal_matching_score = -1
                        raw_data = -1  # fast5_data['Raw']['Reads'][list(fast5_data['Raw']['Reads'].keys())[0]]
                        read_id = -1  # raw_data.attrs['read_id'].decode('UTF-8')
                        raw_signal_length = -1  # raw_data['Signal'].value.shape[0]
                        fastq = -1  # fast5_data['Analyses']['Basecall_1D_000']['BaseCalled_template']['Fastq'].value
                        sequence = -1  # fastq.split('\n')[1]
                        at_content = -1  # sequence.count('A') +  sequence.count('T')
                        gc_content = -1  # sequence.count('G') +  sequence.count('C')
                        # at_content, gc_content = at_content / (at_content + gc_content), gc_content / (at_content + gc_content)
                        sequence_length = -1  # len(sequence)
                        aligned_section_length = -1
                        failed_parsing = True
                        failed_alignment = True
                        mean_q_score = -1
                        reads_data.append(
                            (read_id, raw_signal_length, sequence_length, clipped_start, clipped_end, mapped_start, mapped_end, num_deletions, num_insertions, num_matches, num_mismatches, num_events, signal_matching_score, failed_parsing, failed_alignment, aligned_section_length, at_content, gc_content, mean_q_score)
                        )
                        continue
                    # Parsing: OK
                    # Alignment: OK
                    alignment = template['Alignment']
                    clipped_start = alignment.attrs['clipped_bases_start']
                    clipped_end = alignment.attrs['clipped_bases_end']
                    mapped_start = alignment.attrs['mapped_start']
                    mapped_end = alignment.attrs['mapped_end']
                    num_deletions = alignment.attrs['num_deletions']
                    num_insertions = alignment.attrs['num_insertions']
                    num_matches = alignment.attrs['num_matches']
                    num_mismatches = alignment.attrs['num_mismatches']
                    num_events = mapped_end - mapped_start
                    signal_matching_score = template.attrs['signal_match_score']
                    raw_data = fast5_data['Raw']['Reads'][list(fast5_data['Raw']['Reads'].keys())[0]]
                    read_id = raw_data.attrs['read_id'].decode('UTF-8')
                    raw_signal_length = raw_data['Signal'].value.shape[0]
                    fastq = fast5_data['Analyses']['Basecall_1D_000']['BaseCalled_template']['Fastq'].value
                    sequence = fastq.split('\n')[1]
                    at_content = sequence.count('A') +  sequence.count('T')
                    gc_content = sequence.count('G') +  sequence.count('C')
                    at_content, gc_content = at_content / (at_content + gc_content), gc_content / (at_content + gc_content)
                    sequence_length = len(sequence)
                    events = fast5_data['Analyses']['RawGenomeCorrected_000']['BaseCalled_template']['Events']
                    last_event = events.value.tolist()[-1]
                    aligned_section_length = last_event[-3] + last_event[-2]
                    failed_parsing = False
                    failed_alignment = False
                    # Obtain mean q score
                    try:
                        aligner = mappy.Aligner(reference_file, preset=str('map-ont'), best_n=1)
                        seq_samp_type = tombo_helper.get_seq_sample_type(fast5_data)
                        std_ref = tombo_stats.TomboModel(seq_samp_type=seq_samp_type)
                        map_results = resquiggle.map_read(fast5_data, aligner, std_ref)
                        mean_q_score = map_results.mean_q_score
                    except tombo_helper.TomboError:
                        mean_q_score = -1
                        failed_alignment = True
                reads_data.append(
                    (read_id, raw_signal_length, sequence_length, clipped_start, clipped_end, mapped_start, mapped_end, num_deletions, num_insertions, num_matches, num_mismatches, num_events, signal_matching_score, failed_parsing, failed_alignment, aligned_section_length, at_content, gc_content, mean_q_score)
                )
    columns = ['read_id', 'raw_signal_length', 'sequence_length', 'clipped_start', 'clipped_end', 'mapped_start', 'mapped_end', 'num_deletions', 'num_insertions', 'num_matches', 'num_mismatches', 'num_events', 'signal_matching_score', 'failed_parsing', 'failed_alignment', 'aligned_section_length', 'at_content', 'gc_content', 'mean_q_score']
    metadata_3xr6 = pd.DataFrame(reads_data, columns=columns)
    metadata_3xr6.to_csv('/home/mario/Projects/project_2/statistics/metadata_3xr6.tsv',
        sep='\t', quoting=csv.QUOTE_NONE, index=False)
    t.toc(restart=True)
    # Process the ap dataset
    dataset_folder = "/home/mario/Projects/project_2/databases/working_ap/reads"
    reference_file = dataset_folder + "/" + "reference.fasta"
    reads_data = []
    for flowcell in ['flowcell1','flowcell2','flowcell3', 'flowcell4']:
        flowcell_folder = dataset_folder + '/' + flowcell + '/' + 'single'
        for read_file in os.listdir(flowcell_folder):
            read_name = read_file
            if not read_file.endswith('fast5'):
                continue
            read_file = flowcell_folder + '/' + read_file
            fast5_data = h5py.File(read_file, 'r')
            try:
                template = fast5_data['Analyses']['RawGenomeCorrected_000']['BaseCalled_template']
            except KeyError:
                # Parsing: WRONG
                # Alignment: WRONG
                clipped_start = -1
                clipped_end = -1
                mapped_start = -1
                mapped_end = -1
                num_deletions = -1
                num_insertions = -1
                num_matches = -1
                num_events = -1
                num_mismatches = -1
                signal_matching_score = -1
                raw_data = fast5_data['Raw']['Reads'][list(fast5_data['Raw']['Reads'].keys())[0]]
                read_id = raw_data.attrs['read_id'].decode('UTF-8')
                raw_signal_length = raw_data['Signal'].value.shape[0]
                fastq = fast5_data['Analyses']['Basecall_1D_000']['BaseCalled_template']['Fastq'].value
                sequence = fastq.split('\n')[1]
                at_content = sequence.count('A') +  sequence.count('T')
                gc_content = sequence.count('G') +  sequence.count('C')
                at_content, gc_content = at_content / (at_content + gc_content), gc_content / (at_content + gc_content)
                sequence_length = len(sequence)
                aligned_section_length = -1
                failed_parsing = True
                failed_alignment = True
                mean_q_score = -1
                reads_data.append(
                    (read_id, raw_signal_length, sequence_length, clipped_start, clipped_end, mapped_start, mapped_end, num_deletions, num_insertions, num_matches, num_mismatches, num_events, signal_matching_score, failed_parsing, failed_alignment, aligned_section_length, at_content, gc_content, mean_q_score)
                )
                continue
            # Parsing: OK
            # Alignment: OK
            alignment = template['Alignment']
            clipped_start = alignment.attrs['clipped_bases_start']
            clipped_end = alignment.attrs['clipped_bases_end']
            mapped_start = alignment.attrs['mapped_start']
            mapped_end = alignment.attrs['mapped_end']
            num_deletions = alignment.attrs['num_deletions']
            num_insertions = alignment.attrs['num_insertions']
            num_matches = alignment.attrs['num_matches']
            num_mismatches = alignment.attrs['num_mismatches']
            num_events = mapped_end - mapped_start
            signal_matching_score = template.attrs['signal_match_score']
            raw_data = fast5_data['Raw']['Reads'][list(fast5_data['Raw']['Reads'].keys())[0]]
            read_id = raw_data.attrs['read_id'].decode('UTF-8')
            raw_signal_length = raw_data['Signal'].value.shape[0]
            fastq = fast5_data['Analyses']['Basecall_1D_000']['BaseCalled_template']['Fastq'].value
            sequence = fastq.split('\n')[1]
            at_content = sequence.count('A') +  sequence.count('T')
            gc_content = sequence.count('G') +  sequence.count('C')
            at_content, gc_content = at_content / (at_content + gc_content), gc_content / (at_content + gc_content)
            sequence_length = len(sequence)
            events = fast5_data['Analyses']['RawGenomeCorrected_000']['BaseCalled_template']['Events']
            last_event = events.value.tolist()[-1]
            aligned_section_length = last_event[-3] + last_event[-2]
            failed_parsing = False
            failed_alignment = False
            # Obtain mean q score
            try:
                aligner = mappy.Aligner(reference_file, preset=str('map-ont'), best_n=1)
                seq_samp_type = tombo_helper.get_seq_sample_type(fast5_data)
                std_ref = tombo_stats.TomboModel(seq_samp_type=seq_samp_type)
                map_results = resquiggle.map_read(fast5_data, aligner, std_ref)
                mean_q_score = map_results.mean_q_score
            except tombo_helper.TomboError:
                mean_q_score = -1
                failed_alignment = True
            reads_data.append(
                (read_id, raw_signal_length, sequence_length, clipped_start, clipped_end, mapped_start, mapped_end, num_deletions, num_insertions, num_matches, num_mismatches, num_events, signal_matching_score, failed_parsing, failed_alignment, aligned_section_length, at_content, gc_content, mean_q_score)
            )
    columns = ['read_id', 'raw_signal_length', 'sequence_length', 'clipped_start', 'clipped_end', 'mapped_start', 'mapped_end', 'num_deletions', 'num_insertions', 'num_matches', 'num_mismatches', 'num_events', 'signal_matching_score', 'failed_parsing', 'failed_alignment', 'aligned_section_length', 'at_content', 'gc_content', 'mean_q_score']
    metadata_ap = pd.DataFrame(reads_data, columns=columns)
    metadata_ap.to_csv('/home/mario/Projects/project_2/statistics/metadata_ap.tsv',
        sep='\t', quoting=csv.QUOTE_NONE, index=False)
    t.toc()
