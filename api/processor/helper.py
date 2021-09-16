from api.timer import timed
import datetime
from api.models import LeadPropertyDevice, LeadPropertyTuya, LeadPropertyEquipment, LeadPropertyOnset, GlobalEquipment, LeadPropertyRoom
from app.constant import DASHBOARD_DATASET, DASHBOARD_TABLE_ID
from api import bigquery_client
from app.constant import MEAN_STEP, FACTOR
import json
import pandas as pd
import numpy as np
import pytz
import torch
from torch.utils.data import TensorDataset
from app import current_model
import logging

logger = logging.getLogger(__name__)
logger.setLevel('INFO')

class Helper:

    def __init__(self):
        pass

    def should_continue_logging(self, lead_property_equipments):
        for lead_property_equipment in lead_property_equipments:
            if lead_property_equipment.is_logging == 0:
                return False

            if lead_property_equipment.is_logging == -1:
                lead_property_equipment.is_logging = 0
                lead_property_equipment.save()
                return False

        return True

        # for insertion of metadata
    @timed
    def insert_into_bigquery(self, rows_to_insert, table_id, dataset):
        table_ref = bigquery_client.dataset(dataset).table(table_id)
        table = bigquery_client.get_table(table_ref)

        logger.info("data to be inserted  %s", rows_to_insert)

        if not rows_to_insert:
            return None

        errors = bigquery_client.insert_rows(table, rows_to_insert)

        logger.info("errors  %s", errors)

    @timed
    def create_dataset(self, X, look_back=7):
        # Xs, ys = [], []
        Xs = []

        for i in range(len(X)-look_back):
            #  v = X['mains_power'][i:i+look_back].values
            v = X[i:i+look_back].values
            Xs.append(v)
            # ys.append(dummy_y[i+look_back])

        return np.array(Xs).reshape(len(Xs), look_back, len(X.columns))

    @timed
    def predict(self, data, model):
        data = data['model_data']
        x = self.create_dataset(pd.DataFrame(data), look_back=15)
        tensor_x = torch.tensor(x)
        dataset = TensorDataset(tensor_x)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        labels = np.argmax(model(torch.tensor(dataset.tensors[0].to(device)).float()).cpu().detach(), axis=1)
        labels = pd.Series(labels).value_counts().sort_values(ascending=False)

        if len(labels[labels.index != 2]) == 0:
            return 'no_event'

        else:
            return labels[(labels.index != 2)].index[0]

    @timed
    def classify_appliance(self, appliances, power, device_map):

        result = self.predict(power, current_model)

        if result == 'no_event':
            power = abs(power['event_effect'])

            if appliances == 'all':
                return device_map['name'].iloc[(device_map['mean_power_consumption']-power).abs().argsort()[:1]].values[0]
            else:
                if len(device_map.loc[device_map['name'].isin(appliances)]) == 0:
                    return 0
                else:
                    return device_map.loc[device_map['name'].isin(appliances), 'name'].iloc[(device_map.loc[device_map['name'].isin(appliances), 'mean_power_consumption']-abs(power)).abs().argsort()[:1]].values[0]
        else:
            return 'Refrigerator'

    @timed
    def get_onset_metadata(self, property_id, equipment=None):
        property_devices = LeadPropertyEquipment.query.filter_by(lead_property_id=property_id, is_disabled=False).all()

        # device map
        payload = []

        for device in property_devices:
            payload.append({
                'type': device.type,
                'tuya_id': device.device_id,
                'consumption_category': device.consumption_category,
                'room': LeadPropertyRoom.query.filter_by(id=device.lead_property_room_id).first().name,
                'mean_power_consumption': device.mean_power_consumption,
                'phase': device.phase,
                'name': device.name,
                'equipment_uuid': GlobalEquipment.query.filter_by(id=device.global_equipment_id).first().uuid
            })

        return pd.DataFrame(payload)

    @timed
    def energy_estimation(self, temp, other_original, device_map):
        # logging.info('ACTIVE APPLIANCE', temp)
        # logging.info(temp.active_appliances)
        # logging.info(len(temp.active_appliances))
        logger.info('energy_estimation        temp: \n %s', temp)
        logger.info('energy_estimation        other_original: %s', other_original)
        logger.info('energy_estimation        device_map:  %s', device_map)

        energy_dict={l:0 for l in list({x for l in list(temp.active_appliances) for x in l})}
        prev_energy_dict={}
        energy_dict={l:0 for l in list({x for l in list(temp.active_appliances) for x in l})}
        per_second_energy=[]
        ind=0
        df=pd.DataFrame(columns=energy_dict.keys())

        for i in range(len(temp)):
            prev_energy_dict['other']=other_original

            ### Distribution of Energy between Appliances
            if len(prev_energy_dict)==len(temp.active_appliances.iloc[i]):
                prev_energy_dict[list(prev_energy_dict.keys())[-1]]=temp.mains_power.iloc[i]-sum(list(prev_energy_dict.values())[0:-1])

            ### New Appliance is Active
            elif len(prev_energy_dict)<len(temp.active_appliances.iloc[i]):
                new_appliance=list((set(temp.active_appliances.iloc[i]) - set(prev_energy_dict)))[0]
                prev_energy_dict[new_appliance]=temp.mains_power.iloc[i]-sum(prev_energy_dict.values())

            ### One of the Appliances Deactives
            else:
                removed_appliance=list((set(prev_energy_dict)- set(temp.active_appliances.iloc[i])))[0]
                del prev_energy_dict[removed_appliance]
                prev_energy_dict[list(prev_energy_dict.keys())[-1]]=temp.mains_power.iloc[i]-sum(list(prev_energy_dict.values())[0:-1])

            ### Case When Calculated Equipment Usage > 1.5*Mean Consumption
            excess=0
            negative_energy=[]
            for equipment in prev_energy_dict.keys():
                if equipment=='other':
                    continue
                else:
                    equipment_energy=prev_energy_dict[equipment]
        #                 logging.info(prev_energy_dict)
                    if equipment_energy>0:
                        prev_energy_dict[equipment]=min(equipment_energy,device_map.loc[device_map['name']==equipment,'mean_power_consumption'].values[0]*1.5)
                        excess=excess+equipment_energy-min(equipment_energy,device_map.loc[device_map['name']==equipment,'mean_power_consumption'].values[0]*1.5)
                    else:
                        negative_energy.append(equipment)

            ### Remove Appliance if Calculated energy of appliance is negative
            for negative_equipment in negative_energy:
                del prev_energy_dict[negative_equipment]

            ### Add Excessive Energy to other
            other_original=prev_energy_dict['other']
            prev_energy_dict['other']=prev_energy_dict['other']+excess

            ### Check if Sum of all Appliance Energy> Original, if yes remove from original and subsequent appliances
            for appliance_excess in list(prev_energy_dict):
                if (sum(prev_energy_dict.values())-temp.mains_power.iloc[i])<=.015:
                    break
                else:
                    if appliance_excess=='other':
                        prev_energy_dict[appliance_excess]=max(.015,temp.mains_power.iloc[i]-sum(prev_energy_dict.values()))
                    else:
                        prev_energy_dict[appliance_excess]=max(0,temp.mains_power.iloc[i]-sum(prev_energy_dict.values()))

        #     logging.info(str(temp.index[i])+str(prev_energy_dict)+str(excess))
            for energy_equip in prev_energy_dict.keys():
                energy_dict[energy_equip]=energy_dict[energy_equip]+prev_energy_dict[energy_equip]
            per_second_energy.append(str(prev_energy_dict))
        per_second_energy=pd.DataFrame([eval(i) for i in per_second_energy])
        per_second_energy['logged_time_local']=temp.index

        logger.info('energy_estimation         per_second_energy :  %s', per_second_energy)
        logger.info('energy_estimation         energy dict  %s', energy_dict)

        return per_second_energy,energy_dict

    @timed
    def get_onset_features(self, property_id, start_date, end_date, original_data, additional_columns=[], equipment=None):

        logger.info('get_onset_features       original_data: \n %s', original_data)
        # DOUBT = ? what are these additional_columns
        data = original_data
        history_col_list = None

        # table: map of device id, property id and avg_mains_power, historical_data, active_equipments
        historical_data = '[]'
        if equipment:
            historical_data = equipment.historical_data if equipment.historical_data else '[]'

        logger.info('get_onset_features       historical_data: {0}'.format(len(historical_data)))

        if 'other' in data.columns:
            temp = pd.DataFrame(columns=['logged_time_local', 'mains_power'])
            temp = pd.DataFrame(json.loads(historical_data))
            # need to check - historical data
            logging.info('get_onset_features       in other -- temp: \n %s', temp)
            # logging.info(temp, '>>> temp historical_data')

            if len(temp) > 0:
                temp['logged_time_local'] = [datetime.datetime.utcfromtimestamp(timestamp/1000)-datetime.timedelta(hours=0) for timestamp in temp['logged_time_local']]
                # temp['logged_time_local'] = [datetime.datetime.utcfromtimestamp(timestamp) + datetime.timedelta(seconds=19800) for timestamp in temp['timestamp']]
                df = temp[['logged_time_local', 'mains_power']]
                df = temp.append(data[['logged_time_local', 'mains_power']])
            else:
                df = data[['logged_time_local', 'mains_power']]
            n = len(temp)

            logger.info('get_onset_features       in other -- n: {0}'.format(n))
            logger.info('get_onset_features       in other -- df: \n %s', df)

        else:
            df = pd.DataFrame(columns=['logged_time_local', 'mains_power'])
            df = df.append(pd.DataFrame(json.loads(historical_data)))
            # need to check - historical data
            df['logged_time_local'] = [datetime.datetime.utcfromtimestamp(timestamp/1000)-datetime.timedelta(hours=0) for timestamp in df['logged_time_local']]
            n = len(df)
            data['logged_time_local'] = [pd.to_datetime(timestamp)+datetime.timedelta(hours=5.5) for timestamp in data['logged_on_utc']]
            # need to check - column names : logged_on_utc, active_power_P1, p2, p3
            data['mains_power'] = (data['active_power_P1']+data['active_power_P2']+data['active_power_P3'])/1000
            data['apparent_power']= (data['apparent_power_S1']+data['apparent_power_S2']+data['apparent_power_S3'])/1000
            data = data[['logged_time_local', 'mains_power','apparent_power']]
            data = data.sort_values('logged_time_local').drop_duplicates(subset=['logged_time_local']).set_index('logged_time_local').resample('50ms').interpolate().reset_index()
            # data = data.sort_values('logged_time_local').set_index('logged_time_local').resample('50ms').interpolate().reset_index()
            df = df.append(data)

            logger.info('get_onset_features       in not other -- n: {0}'.format(n))
            logger.info('get_onset_features       in not other -- df: \n %s', df)

        ## n is index to return after calculate feature to remove historical data

        # df = self.add_is_second_data(df, start_date)
        df = df.sort_values('logged_time_local')
        data = data.sort_values('logged_time_local')

        logger.info('get_onset_features           final -- df: \n %s', df)

        FREQUENCY = 20

        df['before_mean'] = df['mains_power'].rolling(MEAN_STEP).mean()
        df['before_max_10'] = df['mains_power'].abs().rolling(MEAN_STEP*FACTOR).max()
        df['before_mean_10'] = df['mains_power'].abs().rolling(MEAN_STEP*FACTOR).mean()
        df['after_mean'] = 0
        df['after_mean_10'] = 0
        df['after_max_10'] = 0
        df['per_change'] = 0
        df['time_spent'] = 0
        df['before_mean_apparent'] = df['apparent_power'].rolling(MEAN_STEP).mean()

        # Features
        per_change = []
        after_mean = []
        after_mean_10 = []
        after_max = []

        for i in range(MEAN_STEP, len(df)-MEAN_STEP):
            change = 100*(df['before_mean'].iloc[i+MEAN_STEP]-df['before_mean'].iloc[i-1])/df['before_mean'].iloc[i+MEAN_STEP]
            per_change.append(change)
            after_mean.append(df['before_mean'].iloc[i+MEAN_STEP])

        for i in range(MEAN_STEP*FACTOR, len(df)-MEAN_STEP*FACTOR):
            after_mean_10.append(df['before_mean_10'].iloc[i+MEAN_STEP*FACTOR])
            after_max.append(df['before_max_10'].iloc[i+MEAN_STEP*FACTOR])

        df['per_change'].iloc[MEAN_STEP:-MEAN_STEP] = per_change
        df['after_mean'].iloc[MEAN_STEP:-MEAN_STEP] = after_mean
        df['after_mean_10'].iloc[MEAN_STEP*FACTOR:-MEAN_STEP*FACTOR] = after_mean_10
        df['after_max_10'].iloc[MEAN_STEP*FACTOR:-MEAN_STEP*FACTOR] = after_max
        df['avg_diff'] = df['after_mean']-df['before_mean']
        df['avg_diff'] = df['after_mean']-df['before_mean']

        #  event logic
        df['events'] = 0
        df.loc[(abs(df['avg_diff']) > .015) & (df['mains_power'] < .10), 'events'] = 1
        df.loc[(abs(df['per_change']) > 10) & (df['mains_power'] > .10), 'events'] = 1
        df = df
        data['event_type'] = 0

        # event smoothening
        df['event_smoothening'] = df['events'].rolling(int(FACTOR/2)).mean()
        df['events'] = [1 if i > .5 else 0 for i in df['event_smoothening']]

        #   transition start=1, transition= 2, transition end=3, steady=0
        df.loc[df['events'] == True, 'event_type'] = 2
        df.loc[df['events'] == False, 'event_type'] = 0

        #  map transition states
        df['event_effect'] = 0
        prev_state = 0
        event_mean_before = 0
        for i in range(1,len(df)):
            if df['events'].iloc[i] == 0:
                if prev_state == 1:
                    df['event_type'].iloc[i-1] = 3
                prev_state = 0
            else:
                if prev_state == 0:
                    df['event_type'].iloc[i] = 1
                prev_state = 1

        #### Observable events for on/off activity
        # print('observable events: ', datetime.datetime.now())
        df['observable_event'] = 0
        df['event_effect'] = 0
        df['transition_length'] = 0
        for i in range(len(df)):
            if df['event_type'].iloc[i] == 1:
                after = 0
                j = i + 5 * frequency
                for j in range(i + 5 * frequency, min(i + 5 * frequency + mean_step * factor, len(df) - 1)):
                    if (df['event_type'].iloc[j] != 1):
                        after = after + df['mains_power'].iloc[j]
                    else:
                        break
                after = after / (j - (i + 5 * frequency - 1))
                before = 0
                k = i - 3 * frequency
                for k in range(i - 3 * frequency, max(i - 3 * frequency - mean_step * factor, 0), -1):
                    if (df['event_type'].iloc[k] != 1):
                        before = before + df['mains_power'].iloc[k]
                    else:
                        break
                before = before / (i - (k + 3 * frequency - 1))
                df['event_effect'].iloc[i + mean_step] = after - before
                df['transition_length'].iloc[i + mean_step] = j - (i + 5 * frequency - 1)

        df.loc[(abs(df['event_effect']) > .010), 'observable_event'] = 1
        df.loc[(df['event_effect'] < 0) & (df['observable_event'] == 1), 'observable_event'] = 2


        model_data=[]

        df['model_data']= 0
        df['model_data']= df['model_data'].astype('object')
        df['before_mean'] = df['mains_power'].rolling(MEAN_STEP).mean().fillna(0)
        df['before_mean_apparent'] = df['apparent_power'].rolling(MEAN_STEP).mean().fillna(0)
        df=df[:6000]
        model_data=[]
        for i in range(n,len(df)):
            temp=pd.DataFrame(np.zeros([100,4]),columns=['mp','ap','nmp','nap'])
            if i<len(df)-80+1:
                temp['mp'] = df.mains_power.iloc[(i-20):(i+80)].values
                temp['ap'] = df.apparent_power.iloc[(i-20):(i+80)].values
                temp['nmp']= df.mains_power.iloc[(i-20):(i+80)].values - df.mains_power.iloc[(i-30):(i-20)].mean()
                temp['nap']= df.apparent_power.iloc[(i-20):(i+80)].values - df.apparent_power.iloc[(i-30):(i-20)].mean()
            else:
                ind=100-(i+80-len(df)+1)
                temp.loc[:ind,'mp'] = df.mains_power.iloc[(i-20):min(i+80,len(df))].values
                temp.loc[ind:,'mp'] = temp.loc[100-(i+80+1-len(df)),'mp']
                temp.loc[:ind,'ap'] = df.apparent_power.iloc[(i-20):min(i+80,len(df))].values
                temp.loc[ind:,'ap'] = temp.loc[100-(i+80+1-len(df)),'ap']
                temp.loc[:ind,'nmp'] = df.mains_power.iloc[(i-20):min(i+80,len(df))].values - df.mains_power.iloc[(i-30):(i-20)].mean()
                temp.loc[ind:,'nmp'] = temp.loc[100-(i+80+1-len(df)),'nmp']
                temp.loc[:ind,'nap'] = df.apparent_power.iloc[(i-20):min(i+80,len(df))].values
                temp.loc[ind:,'nap'] = temp.loc[100-(i+80+1-len(df)),'nap']

            model_data.append(temp.values.tolist())

        df.loc[n:,'model_data']=model_data

        return df[n:]

    @timed
    def get_onset_output(self, property_id, start_date, end_date, temp_data, device_map, equipment=None,
                         history_col_list=['logged_time_local', 'mains_power']):
        logger.info('get_onset_output         start_date: {0}  and end_date: {1}'.format(start_date, end_date))

        logger.info('get_onset_output         temp_data: \n %s', temp_data)

        ind = 1
        onset_device = equipment
        # need to check
        active_names = json.loads(onset_device.active_equipments) if onset_device.active_equipments else {'other': start_date.date()}
        prev_avg_mains_power = onset_device.avg_mains_power if onset_device.avg_mains_power else .088
        # stdate = start_date.replace(tzinfo=None) + datetime.timedelta(seconds=19800)  # local time
        # enddate = end_date.replace(tzinfo=None) + datetime.timedelta(seconds=19800)
        stdate = start_date
        enddate = end_date

        event_df = pd.DataFrame(columns=['start_on_local', 'end_on_local', 'avg_mains_power', 'name', 'status',
                                         'energy_consumption',  'event_on_local',   'event_off_local'])
        main_temp = pd.DataFrame()
        appliance_list = []
        # logging.info('temp columns >>', temp_data.columns)
        # logging.info('basic config')
        # for start, end in zip(pd.date_range(stdate, enddate, freq='5min')[:-1], pd.date_range(stdate, enddate, freq='5min')[1:]):
        start = start_date
        end = end_date

        event_df_temp = pd.DataFrame(columns=['start_on_local', 'end_on_local', 'avg_mains_power', 'name', 'status',
                                              'energy_consumption', 'event_on_local', 'event_off_local'])
        temp = temp_data.set_index('logged_time_local')
        temp = temp[(temp.index <= end) & (temp.index >= start)]
        event_df_temp.loc[ind] = [start, end, None, 'all', 'NA', temp['mains_power'].sum()/3600, None, None]
        temp['active_appliances'] = None

        for name in active_names.keys():
            ind = ind+1
            event_df_temp.loc[ind, 'name'] = name
            event_df_temp.loc[event_df_temp['name'] == name, 'event_on_local'] = str(start)
            event_df_temp.loc[event_df_temp['name'] == name, 'status'] = 'active'

        event_df_temp.loc[event_df_temp['name'] == 'other'] = [start, end, prev_avg_mains_power, 'other', 'NA', 0, str(start), str(end)]

        for i in range(len(temp)):
            temp['active_appliances'].iloc[i] = list(active_names)
            if temp['observable_event'].iloc[i] == 1:

                name = self.classify_appliance('all', temp.iloc[i], device_map)
                if name in event_df_temp['name'].unique():
                    if name in active_names.keys():
                        continue
                    else:
                        active_names[name]=str(temp.index[i])
                        event_df_temp.loc[event_df_temp['name']==name,'status']='active'
                        event_df_temp.loc[event_df_temp['name']==name,'event_on_local']=str(event_df_temp.loc[event_df_temp['name']==name,'event_on_local'].values[0])+' '+str(temp.index[i])
                else:
                    ind=ind+1
                    event_df_temp.loc[ind,'event_on_local']=str(temp.index[i])
                    event_df_temp.loc[ind,'name']=name
                    event_df_temp.loc[ind,'status']='active'
                    active_names[name]=str(temp.index[i])

            elif temp['observable_event'].iloc[i]==2:
                # logging.info('else classiy')
                name=self.classify_appliance(active_names.keys(), temp.iloc[i], device_map)
                if name==0:
                    continue
                if  len(event_df_temp.loc[event_df_temp['name']==name,'event_off_local'].dropna())==0:
                    event_df_temp.loc[event_df_temp['name']==name,'event_off_local']=str(temp.index[i])
                else:
                    event_df_temp.loc[event_df_temp['name']==name,'event_off_local']=str(event_df_temp.loc[event_df_temp['name']==name,'event_off_local'].values[0])+','+str(temp.index[i])
                event_df_temp.loc[event_df_temp['name']==name,'status']='off'
                del active_names[name]
            else:
                continue

        for name in active_names.keys():
            if name=='other':
                continue
            if len(event_df_temp.loc[event_df_temp['name']==name,'event_off_local'].dropna())==0:
                event_df_temp.loc[event_df_temp['name']==name,'event_off_local']=str(temp.index[i])
            else:
                event_df_temp.loc[event_df_temp['name']==name,'event_off_local']=str(event_df_temp.loc[event_df_temp['name']==name,'event_off_local'].values[0])+','+str(temp.index[i])
        event_df_temp['start_on_local']=start

        event_df_temp['end_on_local']=end

        for iterator in range(3):
            if iterator==0:
                logger.info('get_onset_output         in interatore')
                per_second_energy,energy_dict=self.energy_estimation(temp, prev_avg_mains_power, device_map)
            else:
                per_second_energy,energy_dict=self.energy_estimation(sample_df, prev_avg_mains_power, device_map)
            # per_second_energy['is_second'] = temp['is_second']
            per_second_energy['mains_power'] = per_second_energy['other']

            for equip in list(energy_dict):
                if equip=='other':
                    continue
                else:
                    event_df_temp.loc[event_df_temp['name']==equip,'energy_consumption']=event_df_temp.loc[event_df_temp['name']==equip,'energy_consumption'].fillna(0)+energy_dict[equip]/3600
            sample_df = self.get_onset_features(property_id, stdate, enddate, per_second_energy, equipment=equipment)

            # logging.info('sample df >>>>>', sample_df, sample_df.columns)
            sample_df=sample_df.set_index('logged_time_local')
            sample_df['active_appliances']=temp['active_appliances']


        # update historical data after each iteration
        onset_device.historical_data = str(temp.reset_index()[history_col_list][-1200:].to_json(orient='records'))
        logger.info('get_onset_output         updating equipment historical_data %s', onset_device.historical_data)
        onset_device.save()
        logger.info('get_onset_output         updated equipment historical_data')

        event_df_temp.loc[event_df_temp['name']=='other','energy_consumption']=energy_dict['other']/3600
        prev_avg_mains_power=energy_dict['other']/3600
        event_df=event_df.append(event_df_temp)
        main_temp=main_temp.append(temp)

        # converting end time in utc tz aware
        enddate = (end - datetime.timedelta(seconds=19800)).replace(tzinfo=pytz.utc)

        device_map1 = device_map[['name', 'tuya_id']].set_index('name').to_dict()['tuya_id']
        device_map2 = device_map[['name', 'room']].set_index('name').to_dict()['room']
        device_map3 = device_map[['name', 'consumption_category']].set_index('name').to_dict()['consumption_category']
        device_map4 = device_map[['name', 'phase']].set_index('name').to_dict()['phase']
        device_map5 = device_map[['name', 'type']].set_index('name').to_dict()['type']
        device_map6 = device_map[['name', 'equipment_uuid']].set_index('name').to_dict()['equipment_uuid']
        event_df['tuya_id'] = event_df['name'].map(device_map1)
        event_df['room'] = event_df['name'].map(device_map2)
        event_df['consumption_category'] = event_df['name'].map(device_map3)
        event_df['phase'] = event_df['name'].map(device_map4)
        event_df['equipment_type'] = event_df['name'].map(device_map5)
        event_df['equipment_uuid'] = event_df['name'].map(device_map6)
        event_df.loc[event_df['name'] == 'other', 'consumption_category'] = event_df.loc[event_df['name']!='all','consumption_category'].fillna('other')
        event_df.loc[event_df['name'] == 'other', 'type'] = 'other'

        event_df = event_df.rename(columns={'name': 'equipment_name'})

        logger.info('get_onset_output         event_df: \n %s', event_df)
        for equipment in active_names:
            active_names[equipment] = str(active_names[equipment])

        #  updating avg mains power and active equipments in tranasaction db
        onset_device.avg_mains_power = event_df.to_dict()['avg_mains_power'][2]
        onset_device.active_equipments = json.dumps(active_names)

        logger.info('get_onset_output         updating equipment avg_mains_power: {0}'.format(str(onset_device.avg_mains_power)))
        logger.info('get_onset_output         updating equipment active_equipments  {0}'.format(str(onset_device.active_equipments)))

        onset_device.save()
        logger.info('get_onset_output         updated equipment')

        return event_df.fillna(np.nan).replace([np.nan], [None]), enddate
