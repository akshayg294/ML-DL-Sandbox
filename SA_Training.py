import datetime

from SA_TrainingFunctions import *

# all_training_data = pd.read_pickle('Data/logger_data_11_08_2021.pkl')
# labelled_data = label_training_data('devid_C82B9690A288', 1403, all_training_data)
# labelled_data.to_pickle('Data/labelled_data_11_08_2021.pkl')

labelled_data = pd.read_pickle('Data/labelled_data_11_08_2021.pkl')
#
#
labelled_data['mains_power'] = labelled_data['active_power_P1'] + labelled_data['active_power_P2'] + labelled_data[
    'active_power_P3']
labelled_data = labelled_data.sort_values(by=['logged_on_utc', 'created_on_utc'])
labelled_data = labelled_data.set_index('logged_on_utc')

labelled_data_mini = labelled_data[(labelled_data.index >= datetime.datetime(2021, 8, 6)) &
                                   (labelled_data.index <= datetime.datetime(2021, 8, 9))]
labelled_data_mini.to_pickle('Data/labelled_data_060821_090821')

# labelled_data = pd.read_pickle('Data/labelled_data_mini.pkl')

print('done')

# event_instances = labelled_data[labelled_data.label != 'no_event']
#
# # x = get_features(labelled_data)
#
# # utc_start =
#
# event_list = []
#
# for ind in event_instances.index:
#     event_list.append(points_around_each_event(ind, labelled_data))
#     print(ind)

# clean_data(labelled_data)
