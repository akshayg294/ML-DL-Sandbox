import multiprocessing
import numpy as np

import os
import pandas as pd

pd.set_option("display.max_columns", None)

# from keras.utils import np_utils

from tqdm.notebook import tqdm

import warnings

warnings.filterwarnings("ignore")
# from SA_FunctionsRaycon import *
import datetime


def label_data(training_data, labels):
    training_data['label'] = 'no_event'
    for appliance, label, time in zip(labels.equipment_type, labels.event_type, labels.event_time):
        if label == 'ON':
            start_time = time + datetime.timedelta(milliseconds=250)
            end_time = time + datetime.timedelta(milliseconds=750)
        else:
            start_time = time + datetime.timedelta(milliseconds=250)
            end_time = time + datetime.timedelta(milliseconds=750)
        #         else:
        #             start_time = time - datetime.timedelta(milliseconds=200)
        #             end_time = time + datetime.timedelta(milliseconds=200)

        training_data.loc[(training_data.logged_on_utc >= start_time) &
                          (training_data.logged_on_utc <= end_time), 'label'] = appliance + '_' + label

    return training_data


def import_labels(property_id):
    labels = pd.read_excel('Data/Labelled_Data_22_08_2021.xlsx')
    labels = labels[labels['lead_property_id'] == property_id]
    labels['event_time'] = [datetime.datetime.fromtimestamp(timestamp / 1000) - datetime.timedelta(hours=5.5)
                            for timestamp in labels['event_time']]
    return labels


def label_training_data(device_id, property_id, raw_training_data):
    labels = import_labels(property_id)
    training_data = raw_training_data[raw_training_data.device_id == device_id]
    training_data = label_data(training_data, labels)
    return training_data


def resample_data(ld):
    out = ld.set_index('logged_on_local')
    out = out.asfreq(freq='50ms')
    return out


columns = ['voltage_V1', 'voltage_V2', 'voltage_V3', 'current_i1', 'current_i2', 'current_i3', 'power_factor_PF1',
           'power_factor_PF2', 'power_factor_PF3', 'label']
lookback = 16
upahead = 0
nanlimit = 5


def create_training_data(rd):
    print('Process ID: ', os.getpid())
    out = rd[columns]
    n = len(out)
    all_train = []
    all_label = []
    out_labels = out.label.values
    for i in tqdm(range(lookback, n - upahead - 1)):
        chunk = out[i - lookback: i + upahead]
        nanvals = chunk.isnull().values.any()
        if nanvals:
            continue
        #         chunk.interpolate(method='quadratic')
        all_train.append(chunk[columns[:-1]].values)
        all_label.append(out_labels[i + 1])
    return all_train, all_label


if __name__ == '__main__':
    all_training_data = pd.read_pickle('Data/logger_data160821180821.pkl')
    labelled_data = label_training_data('devid_C82B9690A288', 1403, all_training_data)
    labelled_data.to_pickle('Data/labelled_data_22_08_2021.pkl')
    labels = import_labels(1403)
    labels = labels.sort_values(by='event_time')

    # labelled_data = pd.read_pickle('Data/labelled_data_22_08_2021.pkl')

    labelled_data.drop_duplicates(subset='logged_on_local', inplace=True)

    test = resample_data(labelled_data)

    num_processes = multiprocessing.cpu_count()
    chunk_size = int(test.shape[0] / num_processes)
    chunks = [test[i:i + chunk_size + lookback] for i in range(0, test.shape[0], chunk_size)]
    pool = multiprocessing.Pool(processes=num_processes)

    # train, label = create_training_data(test, columns, 15, 0, 5)
    # apply our function to each chunk in the list
    result = pool.map(create_training_data, chunks)
