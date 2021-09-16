from SA_FunctionsRaycon import *

# path = r'C:\Users\Akshay Gupta\Documents\Projects\Homescape\Raycon'
# os.chdir(path)
path = os.getcwd()
print('Path is: ', path)

# # Make Model
# LOOK_BACK = 15
# model = nn.Sequential(
#     Reshape(out_shape=(5, LOOK_BACK)),
#     InceptionBlock(
#         in_channels=5,
#         n_filters=32,
#         kernel_sizes=[5, 13, 23],
#         bottleneck_channels=32,
#         use_residual=True,
#         activation=nn.ReLU()
#     ),
#     InceptionBlock(
#         in_channels=32 * 4,
#         n_filters=32,
#         kernel_sizes=[5, 13, 23],
#         bottleneck_channels=32,
#         use_residual=True,
#         activation=nn.ReLU()
#     ),
#     nn.AdaptiveMaxPool1d(output_size=1),
#     #     nn.AdaptiveMaxPool1d(output_size=1),
#     Flatten(out_features=32 * 4 * 1),
#     nn.Linear(in_features=4 * 32 * 1, out_features=3)
# )
# model.load_state_dict(torch.load(path + '\Source\Inception_time.pth', map_location=torch.device('cpu')))
# print(model.eval())

# raycon_data = getdata(path, False, True)

raycon_data = pd.read_pickle('Data/labelled_data_11_08_2021.pkl')

historical_json = raycon_data.iloc[:60 * 20, :].to_json()
device_map = pd.read_excel(path + '\\Data\\Tuya devices power.xlsx').fillna('none')
stdate = '2021-08-06 00:00:00'
enddate = '2021-08-09 23:59:59'
# pivot_data=getmetadata(pivot_data,device_map)
sample_data = getfeatures(stdate, enddate, raycon_data, historical_json)

print(sample_data['model_data'])

print('script end')

# temp, output = getoutput(stdate, enddate, sample_data, device_map, historical_json)
#
# for ind in sample_data.index[sample_data['observable_event'] == 1]:
#     plot_around_observed_events(ind, sample_data)

