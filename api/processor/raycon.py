import json
import numpy as np
import datetime

import torch
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
from app.constant import DEFAULT_TIMEZONE, DUMP_DATASET, DASHBOARD_TABLE_ID, REPORT_DATASET, PROJECT_ID
import pytz
from api.processor.helper import Helper
from api.models import DisaggregationTrainingData
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from api import bigquery_client, log_error
from api.util import convert_utc_datetime_to_timestamp, get_data_for_disaggregation
from app import current_model as ml_model
import traceback

from api.timer import timed

import logging
logger = logging.getLogger(__name__)
logger.setLevel('INFO')

#%%

LOOK_BACK = 15


class RayconProcessor:

    def __init__(self, property, property_equipments):
        self.LOOK_BACK = LOOK_BACK
        self.property = property
        self.property_id = property.id
        self.property_equipments = property_equipments
        self.timezone = DEFAULT_TIMEZONE
        self.DUMP_DATASET = DUMP_DATASET
        self.REPORT_DATASET = REPORT_DATASET
        self.disaggregation_table_id = 'disaggregation'
        self.TABLE_ID = 'raycon'
        self.disaggregation_table_rows = []
        self.helper = Helper()

    def should_log(self):
        return self.helper.should_continue_logging(self.property_equipments)

    def get_features(self, x):

        x['prev_mean'] = x['mains_power'].rolling(5).mean()
        x['time_since_last_event'] = 0
        x['time_since_event_start'] = 0
        x['normalized_mains_power'] = 0

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

            elif (label != 'no_event') and (activity_ongoing == True):
                x['time_since_last_event'].iloc[i] = (x['logged_on_utc'].iloc[i]-prev_activity_time).microseconds/1000000 + (x['logged_on_utc'].iloc[i]-prev_activity_time).seconds
                x['time_since_event_start'].iloc[i] = (x['logged_on_utc'].iloc[i]-current_activity_time).microseconds/1000000 +(x['logged_on_utc'].iloc[i]-current_activity_time).seconds
                x['normalized_mains_power'].iloc[i] = x['mains_power'].iloc[i]-prev_mean

            elif (label == 'no_event') and (activity_ongoing == False):
                x['time_since_last_event'].iloc[i] = (x['logged_on_utc'].iloc[i]-prev_activity_time).microseconds/1000000 + (x['logged_on_utc'].iloc[i]-prev_activity_time).seconds
                x['time_since_event_start'].iloc[i] = 0
                x['normalized_mains_power'].iloc[i] = x['mains_power'].iloc[i]-prev_mean
        return x

    def label_data(self, training_data, labels):
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

            training_data.loc[(training_data.logged_on_utc >= start_time) & (training_data.logged_on_utc <= end_time), 'label']= appliance+'_'+label

        return training_data

    def import_labels(self, property_id, start_time, end_time):
        #  TO-DO: connect transation db. remove excel link
        # labels = pd.read_excel('labelled_data_07072021.xlsx')

        data_list = DisaggregationTrainingData.query.filter(DisaggregationTrainingData.event_time <= 1614497808800, DisaggregationTrainingData.event_time >= 1614384204400,
                                                            DisaggregationTrainingData.lead_property_id == property_id).all()
        result = []
        for data in data_list:
            result.append({
                'lead_property_id': data.lead_property_id,
                'device_id': data.device_id,
                'activity_type': data.activity_type,
                'equipment_type': data.equipment_type,
                'event_time': data.event_time,
                'event_type': data.event_type
            })
        labels = pd.DataFrame.from_records(result)
        labels = labels[labels['lead_property_id'] == property_id]
        labels['event_time'] = [datetime.datetime.fromtimestamp(timestamp/1000)-datetime.timedelta(hours=5.5) for timestamp in labels['event_time']]
        return labels

    def import_appliance_test_data(self, property_id, appliance, start_time, end_time):
        # doubt: what is this appliance?
        client = bigquery_client
        last_timestamp = '2000-11-11'
        labels = self.import_labels(property_id, convert_utc_datetime_to_timestamp(start_time), convert_utc_datetime_to_timestamp(end_time))

        query = ''
        for time in labels[(labels.event_time > last_timestamp) & (labels.equipment_type == appliance)].event_time[:10]:
            start_time = time - datetime.timedelta(seconds=20)
            end_time = time + datetime.timedelta(seconds=20)
            query = query + 'logged_on_utc >= "' + str(start_time) + '" and logged_on_utc <= "' + str(end_time) + '" or '

        DATASET = PROJECT_ID + '.' + self.DUMP_DATASET + "." + self.TABLE_ID
        dashboard_query = """
                            SELECT
                                *
                            FROM
                                `{0}`
                            WHERE
                                {1}
                            ORDER BY
                                logged_on_utc
                            """.format(DATASET, query[:-4])

        if query == '':
            return None
        else:
            query = 'select * from dev_loggers.raycon where ' + query[:-4] + ' order by logged_on_utc'

            query_job = client.query(dashboard_query)
            results = query_job.result().to_dataframe()
            training_data = results.dropna(subset=['logged_on_utc'])
    #         results['logged_on_utc']=pd.to_datetime(results['logged_on_utc'])
            training_data['logged_on_utc'] = pd.to_datetime(training_data['logged_on_utc'])
    #         training_data=training_data.append(results)
            training_data = training_data.drop_duplicates(subset=['logged_on_utc'])
    #         training_data=training_data.reset_index()
            training_data = self.label_data(training_data, labels)
    #         training_data.to_pickle('raycon_appliance_training_data.pkl')
            return training_data

    def prediction(self, model, start_time=None, end_time=None, property_id=20):
        # TO-DO: Refrigerator?? doubt
        test_data = self.import_appliance_test_data(property_id, 'Refrigerator', start_time, end_time)

        if test_data.empty:
            return False

        test_data = test_data[test_data['label'].isin(['Refrigerator_OFF', 'Refrigerator_ON', 'no_event'])]
        test_data['mains_power'] = test_data['active_power_P1']+test_data['active_power_P2']+test_data['active_power_P3']
        x_test = self.get_features(test_data)
        x_test_values = x_test[['mains_power', 'prev_mean', 'time_since_last_event', 'time_since_event_start', 'normalized_mains_power']]
        #%%

        # encode and transform labels
        encoder = LabelEncoder()
        encoder.fit(test_data['label'])
        encoded_Y = encoder.transform(test_data['label'])

        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y = np_utils.to_categorical(encoded_Y)
        X_Test, Y_Test = self.create_dataset(x_test_values, dummy_y, self.LOOK_BACK)
        tensor_x = torch.Tensor(X_Test)  # transform to torch tensor
        tensor_y = torch.Tensor(Y_Test)

        # scaler = RobustScaler()
        # X_train = scaler.fit_transform(tensor_x)
        # X_test = scaler.transform(tensor_y)

        # x_test, y_test = create_dataset(x_test,LOOK_BACK)

        dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
        trainloader = DataLoader(dataset, batch_size=512)  # create your dataloader

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

        y_pred = [int(x_pred[0][i]) for i in range(len(x_pred[0]))]
        y_true = []
        for i in Y_Test:
            if i[0] == 1:
                y_true.append(0)
            elif i[1] == 1:
                y_true.append(1)
            else:
                y_true.append(2)

        logging.info(f1_score(y_true=y_true, y_pred=y_pred, average="macro"))
        logging.info(accuracy_score(y_true=y_true, y_pred=y_pred))
        cf1 = confusion_matrix(y_true=y_true, y_pred=y_pred) # x_axis = predicted, y_axis = ground_truth
        logging.info(cf1)

        #%%
        return True

    @timed
    def _generate_rows_for_dump_table(self, data, start_timestamp, end_timestamp, equipment):
        """
            start_timestamp : local time
            end_timestamp : local time

            It genreate metadata, feature and output. Then insert into bigquery report db.
        """
        logger.info('_generate_rows_for_dump_table    start_timestamp: {0} and end_timestamp: {1}'.format(start_timestamp, end_timestamp))

        logger.info('_generate_rows_for_dump_table    data: \n %s', data)

        data['dump'] = data.apply(lambda _: _.to_json(), axis=1)

        data['timestamp'] = data['logged_on_utc'].apply(lambda _: _.timestamp(), 4)

        metadata = self.helper.get_onset_metadata(self.property_id)
        logger.info('_generate_rows_for_dump_table    metadata: \n %s', metadata)

        # calculate feature
        features = self.helper.get_onset_features(self.property_id, start_timestamp, end_timestamp, data, equipment=equipment)
        logger.info('_generate_rows_for_dump_table    features: \n %s', features)

        # calculate output
        output, end_date = self.helper.get_onset_output(self.property_id, start_timestamp, end_timestamp, features, metadata,
                                                        equipment=equipment)
        
        logger.info('_generate_rows_for_dump_table    output: \n %s', output) 
        logger.info('_generate_rows_for_dump_table    end_date: %s', end_date)
        

        for index, row in output.iterrows():
            curr_row = row.to_dict()
            curr_row.update({
                'lead_property_id': self.property_id,
                'created_on_utc': datetime.datetime.utcnow(),
                'created_on_local': datetime.datetime.utcnow() + datetime.timedelta(seconds=self.timezone),
            })
            self.disaggregation_table_rows.append(curr_row)

        logger.info('_generate_rows_for_dump_table    inserting into bigquery')
        self.helper.insert_into_bigquery(self.disaggregation_table_rows, self.disaggregation_table_id, self.REPORT_DATASET)

    @timed
    def _process(self, start_timestamp, end_timestamp, property_id, equipment):
        """
            start_timestamp : local time
            end_timestamp : local time
            equipment: instance of LeadPropertyEquipment

            The function fetch data from bigquery for every 5 min interval and then process the data

        """
        logger.info('_process start_timestamp: {0}  and end_timestamp: {1}'.format(start_timestamp, end_timestamp))
        delta = datetime.timedelta(minutes=5)
        while start_timestamp <= end_timestamp:
            end = start_timestamp + delta - datetime.timedelta(microseconds=1)
            data = get_data_for_disaggregation(start_timestamp, end, self.DUMP_DATASET, self.TABLE_ID, self.property_id, equipment)
            if data.empty:
                # return None, start_timestamp
                start_timestamp += delta
                continue
            try:
                self._generate_rows_for_dump_table(data, start_timestamp, end, equipment)
                start_timestamp += delta
            except Exception as e:
                # data.to_csv('data.csv')
                logger.error(traceback.print_exc())
                logger.error(e)
                logger.error('invalid data format received!')
                log_error(traceback.print_exc())
                return None, start_timestamp

        return True, end_timestamp

    @timed
    def process(self):
        """
            Create an array of interval of 1 day from local current datetime and local last_logged_on.
            Then process for each interval
            At last save the last_logged_on time
        """
        current_time = datetime.datetime.utcnow() + datetime.timedelta(seconds=self.timezone)  # local time
        
        for equipment in self.property_equipments:
            if equipment.last_logged_on:
                last_logged_on = equipment.last_logged_on + datetime.timedelta(seconds=self.timezone) # convert into local
            else:
                last_logged_on = current_time - datetime.timedelta(days=1)

            last_historical_time = None
            if equipment.historical_data:
                historical_data = json.loads(equipment.historical_data)
                timestamp = historical_data[-1].get('logged_time_local', None) if historical_data else None # in ms
                last_historical_time = datetime.datetime.utcfromtimestamp(timestamp/1000) + datetime.timedelta(microseconds=1) if timestamp else None

            last_logged_on = max(last_logged_on, last_historical_time) if last_historical_time else last_logged_on
                    
            processing_dates = []
            
            
            # last_logged_on_date = last_logged_on.date() # iterator
            # delta = datetime.timedelta(days=1) # iterate by 1 day

            # breaking the last logged on and current time into window of day intervals
            # while last_logged_on_date <= current_time.date():
            #     if last_logged_on_date == last_logged_on.date():
            #         logger_start_time = last_logged_on
            #         logger_end_time = datetime.datetime.combine(last_logged_on_date, datetime.datetime.max.time())
            #     elif last_logged_on_date == current_time.date():
            #         logger_start_time = datetime.datetime.combine(last_logged_on_date, datetime.datetime.min.time())
            #         logger_end_time = current_time
            #     else:
            #         logger_start_time = datetime.datetime.combine(last_logged_on_date, datetime.datetime.min.time())
            #         logger_end_time = datetime.datetime.combine(last_logged_on_date, datetime.datetime.max.time())

            #     processing_dates.append((logger_start_time, logger_end_time))
            #     last_logged_on_date += delta

            delta = datetime.timedelta(hours=2)
            while last_logged_on <= current_time:
                logger_start_time = last_logged_on
                logger_end_time = last_logged_on + delta - datetime.timedelta(microseconds=1)

                if logger_end_time > current_time:
                    logger_end_time = current_time
                
                last_logged_on += delta
                
                processing_dates.append((logger_start_time, logger_end_time))

            logger.info('process     processing dates:  %s', processing_dates)

            days_count = 0
            data_found = False
            last_seen_on = last_logged_on
            # property id
            property_id = self.property.id
            # data_found = self._process(processing_dates[0][0], processing_dates[1][1], property_id, equipment)
            for date in processing_dates:
                data_found, last_seen_on = self._process(date[0], date[1], property_id, equipment)
                # if not data_found:
                #     # days_count += 1
                #     last_seen_on = date[0] - datetime.timedelta(seconds=self.timezone)

                days_count += 1
                if days_count == 1:
                    break

            ## have to check if last_seen_on is saving as local time or utc time (we need it to be utc)
            equipment.last_logged_on = last_seen_on - datetime.timedelta(seconds=self.timezone) # convert to utc before writing in table
            logger.info('process    updating equipment last_logged_on:  %s', last_seen_on) 
            equipment.save()
            logger.info('process    updated equipment last_logged_on as :  %s', equipment.last_logged_on) 

