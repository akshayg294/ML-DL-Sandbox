from SA_FunctionsRaycon import *

path = os.getcwd()
print('Path is :', path)
logger_query = 'SELECT * FROM `solar-222307.loggers.raycon` '

#  Run getfeatures on logger data
logger_data = getdata_from_logger(path, logger_query, readfromlocal=False)
# logger_data = logger_data.dropna()
# historical_json = logger_data.iloc[:60 * 20, :].to_json()
#
# stdate = '2021-07-20 00:00:00'
# enddate = '2021-08-05 00:00:00'
#
# devicelist = logger_data.device_id.unique()
# for device in devicelist:
#     ldata = logger_data[logger_data.device_id == device]
#     sample_data = getfeatures(stdate, enddate, ldata, historical_json)
#     sample_data.to_pickle('Data/' + str(device) + '.pkl')

# Get all events
# sample_data = pd.read_pickle('Data/devid_98F4AB78A5A8.pkl')
# datalist = []
# path = os.getcwd()
# for ind in sample_data.index[sample_data['observable_event'] == 1]:
#     datalist.append(plot_around_observed_events(path+'//Data//Turn_On_Plots//', ind, sample_data))

# import tslearn
#
# dataarr = np.load('Data/devid_98F4AB78A5A8_arr.npy')
#
#


# from SA_FunctionsRaycon import *
# st = datetime.date(2021, 8, 2)
# et = datetime.date(2021, 8, 21)
# path = os.getcwd()
# query = 'select * from solar-222307.loggers.raycon where logged_on_utc > "{}" and logged_on_utc < "{}"'.format(str(st),
#                                                                                                                str(et))
# logger_data = getdata_from_logger(path, query, False)
# logger_data.device_id.value_counts()
