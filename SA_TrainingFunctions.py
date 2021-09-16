import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from collections import OrderedDict

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler
import os
import pandas as pd

# from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import json

from google.cloud import bigquery
from google.oauth2 import service_account
import datetime
import pickle

from tqdm.notebook import tqdm

import warnings

warnings.filterwarnings("ignore")
# from SA_FunctionsRaycon import *

from Source.inception import Inception, InceptionBlock


class Flatten(nn.Module):
    def __init__(self, out_features):
        super(Flatten, self).__init__()
        self.output_dim = out_features

    def forward(self, x):
        return x.view(-1, self.output_dim)


class Reshape(nn.Module):
    def __init__(self, out_shape):
        super(Reshape, self).__init__()
        self.out_shape = out_shape

    def forward(self, x):
        return x.view(-1, *self.out_shape)


def label_data(training_data, labels):
    training_data['label'] = 'no_event'
    for appliance, label, time in zip(labels.equipment_type, labels.event_type, labels.event_time):
        if label == 'ON':
            start_time = time - datetime.timedelta(milliseconds=100)
            end_time = time + datetime.timedelta(milliseconds=1000)
        else:
            start_time = time - datetime.timedelta(milliseconds=750)
            end_time = time + datetime.timedelta(milliseconds=100)
        #         else:
        #             start_time = time - datetime.timedelta(milliseconds=200)
        #             end_time = time + datetime.timedelta(milliseconds=200)

        training_data.loc[(training_data.logged_on_utc >= start_time) &
                          (training_data.logged_on_utc <= end_time), 'label'] = appliance + '_' + label

    return training_data


def import_labels(property_id):
    labels = pd.read_excel('Data/Labelled_Data_11_08_2021.xlsx')
    labels = labels[labels['lead_property_id'] == property_id]
    labels['event_time'] = [datetime.datetime.fromtimestamp(timestamp / 1000) - datetime.timedelta(hours=5.5)
                            for timestamp in labels['event_time']]
    return labels


def label_training_data(device_id, property_id, raw_training_data):
    labels = import_labels(property_id)
    training_data = raw_training_data[raw_training_data.device_id == device_id]
    training_data = label_data(training_data, labels)
    return training_data


def get_features(x):
    x['prev_mean'] = x['mains_power'].rolling(5).mean()
    x['time_since_last_event'] = 0
    x['time_since_event_start'] = 0
    x['normalized_mains_power'] = 0

    prev_mean = 0
    prev_activity_time = pd.to_datetime('2021-02-27')
    current_activity_time = pd.to_datetime('2021-02-27')
    activity_ongoing = False

    x_dict = x.to_dict('records')

    for row in tqdm(x_dict):

        label = row['label']

        if (label != 'no_event') and (activity_ongoing == False):
            current_activity_time = row['logged_on_utc']
            prev_mean = row['prev_mean']
            activity_ongoing = True

        elif (label == 'no_event') and (activity_ongoing == True):
            prev_activity_time = row['logged_on_utc']
            activity_ongoing = False
            activity_ongoing = False

        elif (label != 'no_event') and (activity_ongoing == True):
            row['time_since_last_event'] = (row['logged_on_utc'] - prev_activity_time).microseconds / 1000000 + \
                                           (row['logged_on_utc'] - prev_activity_time).seconds
            row['time_since_event_start'] = (row['logged_on_utc'] - current_activity_time).microseconds / 1000000 + \
                                            (row['logged_on_utc'] - current_activity_time).seconds
            row['normalized_mains_power'] = row['mains_power'] - prev_mean

        elif (label == 'no_event') and (activity_ongoing == False):
            row['time_since_last_event'] = (row['logged_on_utc'] - prev_activity_time).microseconds / 1000000 + \
                                           (row['logged_on_utc'] - prev_activity_time).seconds
            row['time_since_event_start'] = 0
            row['normalized_mains_power'] = row['mains_power'] - prev_mean
    return x


def points_around_each_event(event_id, labelled_data):
    freq = 20
    secondsbef = 2
    secondsaft = 2
    st_event = max(0, event_id - 20 * 2)
    en_event = min(len(labelled_data), event_id + 20 * 2)
    out = labelled_data[(labelled_data.index >= st_event) & (labelled_data.index <= en_event)]
    out = out.sort_values('logged_on_utc')
    return out


def clean_data(labelled_data):
    # ensure no timestamps are missing in between
    # labelled_data = labelled_data.sort_values(by=['logged_on_utc', 'created_on_utc'])
    # labelled_data = labelled_data.set_index('logged_on_utc')
    data_dups = np.bitwise_not(labelled_data.index.duplicated(keep='last'))
    labelled_data = labelled_data[data_dups]
    labelled_data = labelled_data.resample('50ms').asfreq()
    return labelled_data

