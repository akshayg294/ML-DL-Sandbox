import pickle

from FinalPipelinFunctions import *

if __name__ == '__main__':
    predictor = Predictor()
    predictor.property_id = 1403
    predictor.device_id = 'devid_C82B9690A288'
    predictor.window = 20
    predictor.window_threshold = 10
    predictor.get_data(datapath='Data/logger_data_11_08_2021.pkl')
    predictor.get_labels(datapath='Data/Labelled_Data_22_08_2021.xlsx')
    predictor.get_label_chunks()
    predictor.get_nolabel_chunks()
    # predictor.plot_data()
    predictor.label_finallist = predictor.apply_get_features()
    predictor.nolabel_finallist = predictor.apply_get_features(labelled_bool=False)
    # predictor.label_finallist = pickle.load(open('Data/temp_label_finallist.pkl', 'rb'))
    # predictor.nolabel_finallist = pickle.load(open('Data/temp_nolabel_finallist.pkl', 'rb'))
    predictor.label_train, predictor.label_targets = predictor.create_dataset()
    predictor.nolabel_train, predictor.nolabel_targets = predictor.create_dataset(labelled_bool=False)
    for i in range(0, len(predictor.label_train), 10):
        plot_final_data(predictor.label_train[i], predictor.label_targets[i], 'Data/Predictor_110821/Label_chunks/', i)
    predictor.all_train_test_dataset()
    predictor.create_model()
    predictor.train_model()

    print('### Training Metrics ###')
    predictions = predictor.predict_model(predictor.training_data)
    final_preds = predictor.onehot_to_label(predictions)
    voted_preds = predictor.implement_voting(final_preds)
    voted_labels = predictor.implement_voting(predictor.training_label)
    print("Without Voting")
    predictor.metrics(predictor.training_label, final_preds)
    print("With Voting")
    predictor.metrics(voted_labels, voted_preds)
    print('### Validation Metrics ###')
    predictions = predictor.predict_model(predictor.validation_data)
    final_preds = predictor.onehot_to_label(predictions)
    voted_preds = predictor.implement_voting(final_preds)
    voted_labels = predictor.implement_voting(predictor.validation_label)
    print("Without Voting")
    predictor.metrics(predictor.validation_label, final_preds)
    print("With Voting")
    predictor.metrics(voted_labels, voted_preds)
    print('done')
