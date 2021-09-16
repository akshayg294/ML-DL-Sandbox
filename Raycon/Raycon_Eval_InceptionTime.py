# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 10:23:13 2021

@author: Akshay Gupta
"""

import numpy as np 

import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import os
import pandas as pd

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from google.cloud import bigquery
from google.oauth2 import service_account
import datetime

import warnings
warnings.filterwarnings("ignore")

path = r'C:\Users\Akshay Gupta\Documents\Projects\Homescape\Raycon'
os.chdir(path)

#%%
from inception import InceptionBlock

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

#%%


#### Make Model
LOOK_BACK = 15
model= nn.Sequential(
                    Reshape(out_shape=(5,LOOK_BACK)),
                    InceptionBlock(
                        in_channels=5, 
                        n_filters=32, 
                        kernel_sizes=[5,13,23],
                        bottleneck_channels=32,
                        use_residual=True,
                        activation=nn.ReLU()
                    ),
                    InceptionBlock(
                        in_channels=32*4, 
                        n_filters=32, 
                        kernel_sizes=[5,13,23],
                        bottleneck_channels=32,
                        use_residual=True,
                        activation=nn.ReLU()
                    ),
                    nn.AdaptiveMaxPool1d(output_size=1),
#     nn.AdaptiveMaxPool1d(output_size=1),
                    Flatten(out_features=32*4*1),
                    nn.Linear(in_features=4*32*1, out_features=3)
        )
model.load_state_dict(torch.load(path+'\Inception_time.pth'))
model.eval()

#%%
def get_features(x):
    
    x['prev_mean']=x['mains_power'].rolling(5).mean()
    x['time_since_last_event']=0
    x['time_since_event_start']=0
    x['normalized_mains_power']=0

    prev_mean=0
    prev_activity_time = pd.to_datetime('2021-02-27')
    current_activity_time = pd.to_datetime('2021-02-27')   
    activity_ongoing = False

    for i in range(len(x)):

        label = x['label'].iloc[i]

        if (label != 'no_event') and (activity_ongoing == False):
            current_activity_time = x['logged_on_utc'].iloc[i]
            prev_mean = x['prev_mean'].iloc[i]
            activity_ongoing=True

        elif (label == 'no_event') and (activity_ongoing == True):
            prev_activity_time = x['logged_on_utc'].iloc[i]
            activity_ongoing = False
            activity_ongoing = False

        elif (label!= 'no_event') and (activity_ongoing == True):
            x['time_since_last_event'].iloc[i] = (x['logged_on_utc'].iloc[i]-prev_activity_time).microseconds/1000000 + (x['logged_on_utc'].iloc[i]-prev_activity_time).seconds
            x['time_since_event_start'].iloc[i] = (x['logged_on_utc'].iloc[i]-current_activity_time).microseconds/1000000 +(x['logged_on_utc'].iloc[i]-current_activity_time).seconds  
            x['normalized_mains_power'].iloc[i] = x['mains_power'].iloc[i]-prev_mean

        elif (label== 'no_event') and (activity_ongoing == False):
            x['time_since_last_event'].iloc[i] = (x['logged_on_utc'].iloc[i]-prev_activity_time).microseconds/1000000 + (x['logged_on_utc'].iloc[i]-prev_activity_time).seconds
            x['time_since_event_start'].iloc[i] = 0  
            x['normalized_mains_power'].iloc[i] = x['mains_power'].iloc[i]-prev_mean
    return x

def create_dataset (X, look_back = 7):
    Xs, ys = [], []
 
    for i in range(len(X)-look_back):
#         v = X['mains_power'][i:i+look_back].values
        v = X[i:i+look_back].values
        Xs.append(v)
        ys.append(dummy_y[i+look_back])
 
    return np.array(Xs).reshape(len(Xs), look_back, len(X.columns)), np.array(ys)


def label_data(training_data,labels):
    training_data['label']='no_event'
    for appliance,label,time in zip(labels.equipment_type,labels.event_type,labels.event_time):
        if label=='ON':
            start_time = time - datetime.timedelta(milliseconds=100)
            end_time = time + datetime.timedelta(milliseconds=1000)
        else:
            start_time = time - datetime.timedelta(milliseconds=750)
            end_time = time + datetime.timedelta(milliseconds=100)
#         else:
#             start_time = time - datetime.timedelta(milliseconds=200)
#             end_time = time + datetime.timedelta(milliseconds=200)

        training_data.loc[(training_data.logged_on_utc>=start_time) & (training_data.logged_on_utc<=end_time),'label']= appliance+'_'+label

    return training_data

def import_labels(property_id):
    labels=pd.read_excel('labelled_data_07072021.xlsx')
    labels=labels[labels['lead_property_id']==property_id]
    labels['event_time']=[datetime.datetime.fromtimestamp(timestamp/1000)-datetime.timedelta(hours=5.5) for timestamp in labels['event_time']]
    return labels

def import_appliance_test_data(property_id,appliance):
    os.chdir(path+ r'\Config')
    credentials = service_account.Credentials.from_service_account_file('bigqueryauth.json')
    project_id = 'solar-222307'
    client = bigquery.Client(credentials= credentials,project=project_id)
    
    os.chdir(path)

    last_timestamp='2000-11-11'
    labels=import_labels(property_id)
    
    
    query=''
    for time in labels[(labels.event_time>last_timestamp) & (labels.equipment_type==appliance)].event_time[:10]:
        start_time = time - datetime.timedelta(seconds=20)
        end_time = time + datetime.timedelta(seconds=20)
        query= query + 'logged_on_utc >= "'+ str(start_time) + '" and logged_on_utc <= "' + str(end_time) + '" or '
    
    if query=='':
        return None
    else:
        query = 'select * from dev_loggers.raycon where '+ query[:-4] + ' order by logged_on_utc'
        query_job=client.query(query)       
        results = query_job.result().to_dataframe()
        training_data = results.dropna(subset=['logged_on_utc'])
#         results['logged_on_utc']=pd.to_datetime(results['logged_on_utc'])
        training_data['logged_on_utc']=pd.to_datetime(training_data['logged_on_utc'])
#         training_data=training_data.append(results)
        training_data=training_data.drop_duplicates(subset=['logged_on_utc'])
#         training_data=training_data.reset_index()
        training_data=label_data(training_data,labels)
#         training_data.to_pickle('raycon_appliance_training_data.pkl')
        return training_data

#%%
### Import Data
test_data=import_appliance_test_data(20,'Refrigerator')
test_data=test_data[test_data['label'].isin(['Refrigerator_OFF','Refrigerator_ON','no_event'])]
test_data['mains_power']=test_data['active_power_P1']+test_data['active_power_P2']+test_data['active_power_P3']
x_test=get_features(test_data)
x_test_values=x_test[['mains_power','prev_mean','time_since_last_event','time_since_event_start','normalized_mains_power']]
#%%



# encode and transform labels 
encoder = LabelEncoder()
encoder.fit(test_data['label'])
encoded_Y = encoder.transform(test_data['label'])

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
X_Test, Y_Test = create_dataset(x_test_values,LOOK_BACK)
tensor_x = torch.Tensor(X_Test) # transform to torch tensor
tensor_y = torch.Tensor(Y_Test)

# scaler = RobustScaler()
# X_train = scaler.fit_transform(tensor_x)
# X_test = scaler.transform(tensor_y)

# x_test, y_test = create_dataset(x_test,LOOK_BACK)

dataset = TensorDataset(tensor_x,tensor_y) # create your datset
trainloader = DataLoader(dataset,batch_size=512) # create your dataloader

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
# x_pred=[]
# for i in range(len(dataset)):

x_pred=[]
with torch.no_grad():
    x_pred.append(np.argmax(model(torch.tensor(dataset.tensors[0].to(device)).float()).cpu().detach(), axis=1))
#     for i in range(len(dataset.tensors[0])):
#         x_pred.append(np.argmax(InceptionTime(torch.tensor(dataset.tensors[0][i].to(device)).float()).cpu().detach(), axis=1))
pd.Series(x_pred).unique()

#%%

y_pred=[int(x_pred[0][i]) for i in range(len(x_pred[0]))]
y_true=[]
for i in Y_Test:
    if i[0]==1:
        y_true.append(0)
    elif i[1]==1:
        y_true.append(1)
    else:
        y_true.append(2)
        
print(f1_score(y_true=y_true, y_pred=y_pred,average="macro"))
print(accuracy_score(y_true=y_true, y_pred=y_pred))
cf1 = confusion_matrix(y_true=y_true, y_pred=y_pred) # x_axis = predicted, y_axis = ground_truth
print(cf1)

#%%