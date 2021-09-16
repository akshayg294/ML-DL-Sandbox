from FinalPipelinFunctions import *

if __name__ == '__main__':
    pred = Predictor()
    pred.property_id = 10413
    pred.tuya_device_id = '7200407140f520ecefa3'
    pred.device_id = 'devid_C82B96932498'
    pred.get_tuya_data(datapath='Data/tuya_data_2021-09-13.pkl')
    # pred.get_data(datapath='Data/raw_data_10413_2021-09-13.pkl')