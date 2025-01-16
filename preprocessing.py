# Imports
import os
import pickle
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# Prespecifications
ids = []
sampling_rate = 100  # Resampling (Hz)
input_length = [30]  # Input data length (sec)
# Prediction lag for 300, 600, and 900 sec (5-, 10-, 15-min prediction)
pred_lag = [300, 600, 900]
pred_threshold = {'hypo': lambda x: x < 65,  # Threshold for hypotension (mmHg)
                  'normo': lambda x: x >= 65}
pred_min_length = {'hypo': 60,  # Minimum duration (sec) for defining hypotensive event
                   'normo': 1200}  # for non-hypotensive event (normal)
convert_dir = './converted/'  # Input path
processed_dir = './processed_short/'  # Output path

if not (os.path.isdir(processed_dir)):
    os.makedirs(os.path.join(processed_dir))

for caseid in ids:
    converted_path = convert_dir+str(caseid)+'.pkl'
    processed_path = processed_dir+'{}.pkl'.format(caseid)
    source = {}
    source_end = 0

    # Read time-synchronized raw data
    if os.path.exists(converted_path) == True:
        with open(converted_path, 'rb') as handle:
            data = pickle.load(handle)

        if len(data.keys()) != 4:
            print(f'Wrong number of columns in {converted_path} !!!')
            exit(1)

        for full_name, track in [['SNUADC/ART', 'abp'],
                                 ['SNUADC/ECG_II', 'ecg'],
                                 ['SNUADC/PLETH', 'ple'],
                                 ['Primus/CO2', 'co2']]:
            source[track] = np.array(data[full_name])
            source_end = max(source_end, source[track].shape[0])

    valid = {}

    # Parameters for peak detection
    peak_height = {'abp': 30, 'ecg': -999, 'ple': 10, 'co2': 20}
    peak_prominence = {'abp': 20, 'ecg': 0.2, 'ple': 10, 'co2': 20}
    peak_distance = {'abp': 0.5, 'ecg': 0.5, 'ple': 0.5, 'co2': 1}

    tolerance_interval = {'abp': 3, 'ecg': 3, 'ple': 3, 'co2': 8}
    tolerance_min = {'abp': 0, 'ecg': -1, 'ple': -10, 'co2': 0}
    tolerance_max = {'abp': 250, 'ecg': 1, 'ple': 100, 'co2': 70}

    peaks = {}
    next_peaks = {}
    # Define rhythm segments based on detected peaks
    for track in source.keys():
        valid[track] = np.ones(source_end)
        valid[track][np.where(source[track] < tolerance_min[track])] = 0
        valid[track][np.where(source[track] > tolerance_max[track])] = 0

        peaks[track] = find_peaks(source[track], distance=peak_distance[track] * sampling_rate,
                                  height=peak_height[track],
                                  prominence=peak_prominence[track])[0]
        next_peaks[track] = np.append(peaks[track][1:], source_end)

        invalid_peaks = np.where(
            ((next_peaks[track] - peaks[track]) <= tolerance_interval[track] * sampling_rate) == False)[0]

        if track == 'abp':
            for x in invalid_peaks:
                valid[track][peaks[track][x]: next_peaks[track][x]] = 0
    n_peaks = len(peaks['abp'])

    try:
        peak_map = np.zeros(n_peaks-1)
    except Exception as e:
        continue

    for i in range(n_peaks-1):
        peak_map[i] = np.mean(source['abp'][peaks['abp'][i]:peaks['abp'][i+1]])
        # Exclude rhythms with abnormal arterial pressure
        if peak_map[i] < 20 or peak_map[i] > 200:
            valid['abp'][peaks['abp'][i]: next_peaks['abp'][i]] = 0

    valid_peak_num = {}
    consecutive_peak_num = {}
    section_stack = {}

    for phase in ['hypo', 'normo']:
        # Extractt peaks based on hypotension threshold
        valid_peak_num[phase] = np.where(pred_threshold[phase](peak_map))[0]

        # Mark -1 for abnormal peaks
        for i, x in enumerate(valid_peak_num[phase]):
            if np.all(valid['abp'][peaks['abp'][x]:peaks['abp'][x+1]]):
                pass
            else:
                valid_peak_num[phase][i] = -1

        # Exclude abnormal peaks
        valid_peak_num[phase] = valid_peak_num[phase][valid_peak_num[phase] != -1]

        # Concatenate consecutive rhythms
        consecutive_peak_num[phase] = np.split(
            valid_peak_num[phase], np.where(np.diff(valid_peak_num[phase]) != 1)[0]+1)

        section_stack[phase] = []
        for i, x in enumerate(consecutive_peak_num[phase]):
            if len(x) > 1:
                start = peaks['abp'][x[0]]
                end = peaks['abp'][x[-1]+1]

                if end - start >= pred_min_length[phase] * sampling_rate:
                    section_stack[phase].append([start, end])

    output = {'input_length': [],
              'abp': [],
              'ecg': [],
              'ple': [],
              'co2': [],
              'pred_lag': [],
              'hypo': [],
              'map': []}

    # Establish dataset

    for phase in ['hypo', 'normo']:
        for pred_start, pred_end in section_stack[phase]:
            for length in input_length:
                for lag in pred_lag:
                    s_end = pred_start - lag * sampling_rate
                    if phase == 'hypo':
                        s_end = pred_start - lag * sampling_rate
                    else:
                        s_end = int((pred_start + pred_end) *
                                    0.5 - (lag * 0.5 * sampling_rate))
                    s_start = s_end - length * sampling_rate
                    for multi in range(2):
                        if phase == 'hypo' and multi >= 1:
                            break
                        section_start = s_start + multi * 10
                        section_end = s_end + multi * 10
                        if section_start >= 0 and section_end <= source_end:
                            output['input_length'].append(length)
                            for track in ['abp', 'ecg', 'ple', 'co2']:
                                output[track].append(
                                    source[track][section_start:section_end])
                            output['pred_lag'].append(lag)
                            output['hypo'].append(1 if phase == 'hypo' else 0)
                            output['map'].append(
                                np.mean(source['abp'][pred_start:pred_end]))

    # Output pre-processed data

    with open(processed_path, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # hypo = pd.DataFrame ( { 'section': section_stack['hypo'] } )
    # hypo['phase'] = 'hypo'

    # normo = pd.DataFrame ( { 'section': section_stack['normo'] } )
    # normo['phase'] = 'normo'

    # section = pd.concat ( [ hypo, normo ], ignore_index=True )
    # section['caseid'] = caseid

    # section_dir = './section/'

    # if not (os.path.isdir(section_dir)):
    #     os.makedirs(os.path.join(section_dir))

    # section.to_csv(section_dir + str(caseid) + '.csv')
