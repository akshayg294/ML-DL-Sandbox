import os
import numpy as np

np.random.seed(1234)
import pandas as pd
import warnings

import random

random.seed(1234)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

torch.manual_seed(1234)
torch.backends.cudnn.benchmark = False

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder

# from keras.utils import np_utils
import json

from google.cloud import bigquery
# from google.cloud import bigquery_storage
from google.oauth2 import service_account
import datetime
import pickle

from tqdm import tqdm

from Source.inception import Inception, InceptionBlock
from SA_FunctionsRaycon import getfeatures


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


# Source: https://stackoverflow.com/questions/50125844/how-to-standard-scale-a-3d-matrix/50134698
class NDStandardScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X


def plot_final_data(data, target, savepath, ind):
    plt.plot(data[:, 0], color='blue')
    plt.plot(data[:, 2], color='red')
    # plt.show()
    plt.savefig(savepath + target + str(ind) + 'active.png')
    plt.close()
    plt.plot(data[:, 1], color='black')
    plt.plot(data[:, 3], color='green')
    # plt.show()
    plt.savefig(savepath + target + str(ind) + 'apparent.png')
    plt.close()
    return


def plot_candlestick(data, xaxis):
    #  Data is an array of shape (len(xaxis), -1)
    avgdata = np.empty(data.shape[0])
    for i in range(len(xaxis)):
        ma = max(data[i, :])
        mi = min(data[i, :])
        avgdata[i] = np.average(data[i, :])
        plt.vlines(xaxis[i], mi, ma)
    plt.plot(xaxis, avgdata, marker='o')
    plt.show()
    return


def converttoint(x):
    try:
        return int(x)
    except ValueError:
        return np.nan


class ExperimentTracker:
    def __init__(self):
        self.lookback = None
        self.learningrate = None
        self.dropout = None
        self.weight = None
        self.window = None
        self.window_thresh = None
        self.batch_size = None
        self.val_voted_preds = None
        self.val_voted_labels = None
        self.tra_voted_preds = None
        self.tra_voted_labels = None


class Predictor:
    def __init__(self):
        # Final Data used for training the model
        self.label_train = None
        self.label_targets = None
        self.nolabel_train = None
        self.nolabel_targets = None
        self.all_data = None
        self.all_label = None
        self.training_data = None
        self.training_label = None
        self.validation_data = None
        self.validation_label = None

        # Model parameters
        self.compute_device = 'cuda'
        self.inception_model = None
        self.inception_kernel_sizes = [5, 13, 29]
        self.inception_bottleneck_channels = 32
        self.lookback = 80
        self.batchsize = 512
        self.epochs = 40
        self.learning_rate = .002
        self.dropout = None
        self.label_weights = None
        self.train_data_scaler = None
        self.label_encoder = None

        # Data parameters for getting data from bigquery
        self.start_date = None
        self.end_data = None
        self.property_id = 1403
        self.device_id = 'devid_C82B9690A288'
        self.tuya_device_id = None
        self.tuya_appliance_mapping = {'7200407140f520ecefa3': 'AC'}
        self.window = 20  # Number of continous samples labelled to a particular target
        self.window_threshold = 5  # Minimum number of samples for a label to be marked
        self.bqcredentials = service_account.Credentials.from_service_account_file('Source/Config/bigqueryauth.json')
        self.bqproject_id = 'solar-222307'

        # Raw data from bigquery, labelled data
        self.labels = None
        self.bigquery_data = None
        self.tuya_data = None
        self.label_data_chunks = None
        self.no_label_data_chunks = None

        # Data Chunks of each individual event
        self.nolabel_samples = 10  # Increase this to increase the number of no_events in training set
        self.nolabel_mintime = 100  # Smallest period of window in seconds between successive events from which to
        # get samples from
        self.label_datalist = None
        self.label_labellist = None
        self.label_timelist = None
        self.nolabel_datalist = None
        self.nolabel_labellist = None
        self.nolabel_timelist = None
        self.tuya_label_datalist = None
        self.tuya_label_labellist = None
        self.tuya_label_timelist = None

        # Data chunks after running getfeatures
        self.label_finallist = None
        self.nolabel_finallist = None

    def get_data(self, read_from_local=True, datapath=None):
        if read_from_local:
            raw_training_data = pd.read_pickle(datapath)
        else:
            print("Getting data from BigQuery")
            client = bigquery.Client(credentials=self.bqcredentials, project=self.bqproject_id)
            query = 'SELECT * FROM `solar-222307.loggers.raycon` where property_id = ' + str(self.property_id)
            query_job = client.query(query)
            raw_training_data = query_job.result().to_dataframe()
            datestr = str(datetime.datetime.now().date())
            raw_training_data.to_pickle('Data/raw_data_' + str(self.property_id) + '_' + datestr + '.pkl')
        training_data = raw_training_data[raw_training_data.device_id == self.device_id]
        training_data = training_data.drop(columns=['dump', 'created_on_utc', 'property_id', 'device_id'])
        self.bigquery_data = training_data.sort_values(by='logged_on_utc')
        return

    def get_labels(self, read_from_local=True, datapath=None):
        if read_from_local:
            labels = pd.read_excel(datapath)
        else:
            print("Not implemented")
            return
        labels = labels[labels['lead_property_id'] == self.property_id]
        labels['event_time'] = [datetime.datetime.utcfromtimestamp(timestamp / 1000)
                                for timestamp in labels['event_time']]
        labels = labels.sort_values(by='event_time')
        self.labels = labels
        return

    def get_tuya_data(self, read_from_local=True, datapath=None):
        if self.property_id == 10413:
            propertyid = 14730
        else:
            propertyid = self.property_id
        if read_from_local:
            tuya_data = pd.read_pickle(datapath)
        else:
            print("Getting data from BigQuery")
            client = bigquery.Client(credentials=self.bqcredentials, project=self.bqproject_id)
            query = 'SELECT * FROM `solar-222307.loggers.tuya`'
            query_job = client.query(query)
            tuya_data = query_job.result().to_dataframe()
            datestr = str(datetime.datetime.now().date())
            tuya_data.to_pickle('Data/tuya_data_' + datestr + '.pkl')
        tuya_data = tuya_data[tuya_data['property_id'] == propertyid]
        tuya_data = tuya_data[tuya_data['device_id'] == self.tuya_device_id]
        tuya_data.value = tuya_data.apply(lambda row: converttoint(row.value), axis=1)
        tuya_data['event_time_local'] = [
            datetime.datetime.utcfromtimestamp(timestamp / 1000) + datetime.timedelta(hours=5.5)
            for timestamp in tuya_data['event_timestamp']]
        tuya_data = tuya_data.sort_values(by='event_time_local')
        self.tuya_data = tuya_data
        return

    def get_label_chunks(self):
        datalist = list()
        labellist = list()
        timelist = list()
        for appliance, label, tim in tqdm(
                zip(self.labels.equipment_type, self.labels.event_type, self.labels.event_time)):
            st = tim - datetime.timedelta(minutes=2)
            et = tim + datetime.timedelta(minutes=1)
            out = self.bigquery_data[
                (st <= self.bigquery_data.logged_on_utc) & (self.bigquery_data.logged_on_utc <= et)]
            if out.empty:
                continue
            datalist.append(out)
            labellist.append(appliance + '_' + label)
            timelist.append(tim + datetime.timedelta(hours=5, minutes=30))
        self.label_datalist = datalist
        self.label_labellist = labellist
        self.label_timelist = timelist
        return

    def get_label_tuya(self):
        datalist = list()
        labellist = list()
        timelist = list()
        for device_id, tim in tqdm(
                zip(self.tuya_data.device_id, self.tuya_data.event_time_utc)):
            st = tim - datetime.timedelta(minutes=2)
            et = tim + datetime.timedelta(minutes=1)
            out = self.bigquery_data[
                (st <= self.bigquery_data.logged_on_utc) & (self.bigquery_data.logged_on_utc <= et)]
            if out.empty:
                continue
            datalist.append(out)
            appliance = self.tuya_appliance_mapping[device_id]
            labellist.append(appliance + '_')
            timelist.append(tim + datetime.timedelta(hours=5, minutes=30))
        self.tuya_label_datalist = datalist
        self.tuya_label_labellist = labellist
        self.tuya_label_timelist = timelist
        return

    def get_no_label_time(self, samples):
        eventlist = self.labels.event_time.tolist()
        starttime = min(self.bigquery_data.logged_on_utc)
        endtime = max(self.bigquery_data.logged_on_utc)
        shorteventlist = [e for e in eventlist if starttime < e < endtime]
        nolabellist = []
        for i in range(len(shorteventlist) - 1):
            st = shorteventlist[i] + datetime.timedelta(seconds=5)
            et = shorteventlist[i + 1] - datetime.timedelta(seconds=5)
            delta = (et - st).total_seconds()
            if delta > 100:
                randsamp = np.random.uniform(0, 1, size=samples)
                randsamp = [st + datetime.timedelta(seconds=delta * r) for r in randsamp]
                for r in randsamp:
                    nolabellist.append(r)
        return nolabellist

    def get_nolabel_chunks(self):
        nolabels = self.get_no_label_time(self.nolabel_samples)
        datalist = list()
        labellist = list()
        timelist = list()
        for nl in tqdm(nolabels):
            st = nl - datetime.timedelta(minutes=2)
            et = nl + datetime.timedelta(minutes=1)
            out = self.bigquery_data[(st <= self.bigquery_data.logged_on_utc) &
                                     (self.bigquery_data.logged_on_utc <= et)]
            if out.empty:
                continue
            datalist.append(out)
            labellist.append('no_event')
            timelist.append(nl + datetime.timedelta(hours=5, minutes=30))
        self.nolabel_datalist = datalist
        self.nolabel_labellist = labellist
        self.nolabel_timelist = timelist
        return

    def plot_data(self, path, labelled_bool=True):
        if labelled_bool:
            datalist, labelist, timelist = self.label_datalist, self.label_labellist, self.label_timelist
        else:
            datalist, labelist, timelist = self.nolabel_datalist, self.nolabel_labellist, self.nolabel_timelist
        ind = 0
        for data, label, tim in zip(datalist, labelist, timelist):
            print(label, tim)
            plt.plot(data.logged_on_utc, data.active_power_P1 + data.active_power_P2 + data.active_power_P3,
                     color='red')
            plt.plot(data.logged_on_utc, data.apparent_power_S1 + data.apparent_power_S2 + data.apparent_power_S3,
                     color='blue')
            plt.savefig(path + label + str(ind) + '.png')
            plt.close()
            ind += 1
        return

    def apply_get_features(self, labelled_bool=True):
        finallist = list()
        if labelled_bool:
            datalist, labelist, timelist = self.label_datalist, self.label_labellist, self.label_timelist
        else:
            datalist, labelist, timelist = self.nolabel_datalist, self.nolabel_labellist, self.nolabel_timelist
        for dat, lab, tim in zip(datalist, labelist, timelist):
            try:
                historical_json = dat.iloc[:60 * 20, :].to_json()
                stdate = str(min(dat.logged_on_local))
                enddate = str(max(dat.logged_on_local))
                out = getfeatures(stdate, enddate, dat, historical_json)
                obs_event = out.logged_on_local[out.observable_event > 0]
                out['label'] = 'no_event'
                selected_index = obs_event.index[np.argmin([abs(o - tim) for o in obs_event])]
                if lab.split('_')[1] == 0:
                    on_off = 'ON' if out.observable_event[selected_index] == 1 else 'OFF'
                    lab = lab + on_off
                out.loc[selected_index, 'label'] = lab
                finallist.append(out)
            except:
                # print(tim, lab)
                pass
        return finallist

    def create_dataset(self, labelled_bool=True):
        if labelled_bool:
            finallist = self.label_finallist
        else:
            finallist = self.nolabel_finallist
        ldataset = list()
        ltargetset = list()

        for i in range(len(finallist)):
            out = finallist[i]
            if labelled_bool:
                ind = out.index[out.label != 'no_event'].tolist()[0]
            else:
                ind = 2000
            out['apparent_power'] = (out.apparent_power_S1 + out.apparent_power_S2 + out.apparent_power_S3) / 1000
            out['mean_app_power'] = out.apparent_power.rolling(30).mean()
            for lab_ind in range(ind - self.window, ind):
                data = out[lab_ind - self.window:lab_ind + (self.lookback - self.window)]
                data['norm_main_power'] = data.mains_power - data.before_mean.tolist()[0]
                data['norm_apparent_power'] = data.apparent_power - data.mean_app_power.tolist()[0]
                data = data[['mains_power', 'apparent_power', 'norm_main_power', 'norm_apparent_power']].values
                ldataset.append(data)
                ltargetset.append(out.label[ind])
        return ldataset, ltargetset

    def implement_voting(self, predictions):
        assert len(predictions) % self.window == 0
        out = list()
        for i in range(0, len(predictions), self.window):
            preddata = predictions[i:i + self.window]
            value_counts = pd.Series(preddata).value_counts()
            if value_counts.index[0] != 'no_event':
                out.append(value_counts.index[0][0])
            elif len(value_counts) > 1:
                if value_counts[1] > self.window_threshold:
                    out.append(value_counts.index[1][0])
            else:
                out.append(value_counts.index[0][0])
        return out

    def all_train_test_dataset(self, train_split=.75):
        self.all_data = np.concatenate([self.label_train, self.nolabel_train])
        self.all_label = np.concatenate([self.label_targets, self.nolabel_targets])

        self.training_data = np.concatenate([
            self.label_train[:int(int(len(self.label_train) / self.window) * train_split) * self.window],
            self.nolabel_train[:int(int(len(self.nolabel_train) / self.window) * train_split) * self.window]])

        self.training_label = np.concatenate([
            self.label_targets[:int(int(len(self.label_train) / self.window) * train_split) * self.window],
            self.nolabel_targets[:int(int(len(self.nolabel_train) / self.window) * train_split) * self.window]])

        self.validation_data = np.concatenate([
            self.label_train[int(int(len(self.label_train) / self.window) * train_split) * self.window:],
            self.nolabel_train[int(int(len(self.nolabel_train) / self.window) * train_split) * self.window:]])

        self.validation_label = np.concatenate([
            self.label_targets[int(int(len(self.label_train) / self.window) * train_split) * self.window:],
            self.nolabel_targets[int(int(len(self.nolabel_train) / self.window) * train_split) * self.window:]])
        return

    def create_model(self, useweights=False):
        self.train_data_scaler = NDStandardScaler()
        self.train_data_scaler.fit(self.training_data)

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.all_label)

        if useweights:
            label_value_counts = pd.DataFrame(self.all_label).value_counts()
            index = [x[0] for x in label_value_counts.index.tolist()]
            values = label_value_counts.values
            encoded_index = self.label_encoder.transform(index)
            values = values[encoded_index]
            sum_values = sum(values)
            self.label_weights = [sum_values / x for x in values]
        else:
            self.label_weights = np.ones(len(self.label_encoder.classes_))

        self.inception_model = nn.Sequential(
            Reshape(out_shape=(4, self.lookback)),
            InceptionBlock(
                in_channels=4,
                n_filters=32,
                kernel_sizes=self.inception_kernel_sizes,
                bottleneck_channels=self.inception_bottleneck_channels,
                use_residual=True,
                activation=nn.ReLU()
            ),
            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=self.inception_kernel_sizes,
                bottleneck_channels=self.inception_bottleneck_channels,
                use_residual=True,
                activation=nn.ReLU()
            ),
            nn.AdaptiveMaxPool1d(output_size=1),
            #     nn.AdaptiveMaxPool1d(output_size=1),
            Flatten(out_features=32 * 4 * 1),
            nn.Linear(in_features=4 * 32 * 1, out_features=len(self.label_encoder.classes_))
        )
        return

    def label_to_onehot(self, labels):
        encoded = self.label_encoder.transform(labels)
        out = np.zeros((encoded.size, encoded.max() + 1))
        out[np.arange(encoded.size), encoded] = 1
        return out

    def onehot_to_label(self, encoded):
        out = np.argmax(encoded, axis=1)
        out = self.label_encoder.inverse_transform(out.squeeze())
        return out

    def train_model(self, all_data_bool=False):
        if all_data_bool:
            tensor_x = torch.Tensor(self.train_data_scaler.transform(self.all_data))
            tensor_y = torch.Tensor(self.label_to_onehot(self.all_label))
        else:
            tensor_x = torch.Tensor(self.train_data_scaler.transform(self.training_data))
            tensor_y = torch.Tensor(self.label_to_onehot(self.training_label))

        dataset = TensorDataset(tensor_x, tensor_y)
        trainloader = DataLoader(dataset, batch_size=self.batchsize, shuffle=True)

        print('Start Trianing')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached: ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

        self.inception_model.to(self.compute_device)
        # print(self.compute_device)

        optimizer = optim.Adam(self.inception_model.parameters(), lr=self.learning_rate)
        weight = torch.Tensor(self.label_weights)
        weight = weight.to(self.compute_device)
        # print("Starting Training")

        criterion = nn.BCEWithLogitsLoss()
        for epoch in range(self.epochs):
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.compute_device), labels.to(self.compute_device)

                optimizer.zero_grad()
                outputs = self.inception_model(inputs)
                loss = criterion(outputs, labels)
                loss = loss * weight
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                # print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
        print('End Trianing. Emptying Cache')
        torch.cuda.empty_cache()
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached: ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
        return

    def deployment_predict(self, input_data):
        assert len(input_data) == self.lookback + self.window, "input data should be of length: " + \
                                                               str(self.lookback + self.window)
        pred_inputs = np.empty((self.window, self.lookback, input_data.shape[1]))
        for i in range(self.window):
            pred_inputs[i] = input_data[i:i+self.lookback]

        preds = self.predict_model(pred_inputs)
        final_preds = self.onehot_to_label(preds)
        voted_preds = self.implement_voting(final_preds)
        return voted_preds

    def predict_model(self, data):  # Output is one hot encoded. Use onehot_to_label to get actual labels
        output = []
        for i in range(0, len(data), self.batchsize):
            tensor_x = torch.Tensor(self.train_data_scaler.transform(data[i:i + self.batchsize]))
            inputs = tensor_x.to(self.compute_device)
            outputs = self.inception_model(inputs)
            output.append(outputs.detach().cpu().numpy())
        return np.concatenate(output)

    def metrics(self, truelabels, predictions):  # truelabels and predictions should not be one hot encoded
        print(f1_score(y_true=truelabels, y_pred=predictions, average="macro"))
        print(accuracy_score(y_true=truelabels, y_pred=predictions))
        cf = confusion_matrix(y_true=truelabels, y_pred=predictions)
        print(cf)
        for i in range(len(cf)):
            print('For label ', self.label_encoder.classes_[i])
            true_positive = cf[i, i]
            false_positive = sum(cf[:, i]) - true_positive
            false_negative = sum(cf[i, :]) - cf[i, i]
            true_negative = sum(sum(cf)) - false_positive - false_negative - true_positive
            print('tp: ', true_positive, ' fp: ', false_positive, ' fn: ', false_negative, ' tn: ', true_negative)
        return
