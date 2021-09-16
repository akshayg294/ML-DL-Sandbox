from FinalPipelinFunctions import *

if __name__ == '__main__':
    # predictor1 = Predictor()
    # predictor1.property_id = 1403
    # predictor1.device_id = 'devid_C82B9690A288'
    # predictor1.window = 20
    # predictor1.window_threshold = 10
    # predictor1.get_data(datapath='Data/logger_data160821180821.pkl')
    # predictor1.get_labels(datapath='Data/Labelled_Data_22_08_2021.xlsx')
    # predictor1.get_label_chunks()
    # predictor1.get_nolabel_chunks()
    # predictor1.label_finallist = predictor1.apply_get_features()
    # predictor1.nolabel_finallist = predictor1.apply_get_features(labelled_bool=False)
    # predictor1.label_train, predictor1.label_targets = predictor1.create_dataset()
    # predictor1.nolabel_train, predictor1.nolabel_targets = predictor1.create_dataset(labelled_bool=False)
    # predictor1.all_train_test_dataset()
    # predictor1.bigquery_data = None
    # with open('Data/predictor_180821_L25.pkl', 'wb') as file:
    #     pickle.dump(predictor1, file)
    predictor1 = pickle.load(open('Data/predictor_180821_L25.pkl', 'rb'))
    predictor2 = pickle.load(open('Data/predictor_110821_L25.pkl', 'rb'))

    fpredictor = Predictor()
    fpredictor.training_data = np.concatenate([predictor1.training_data, predictor2.training_data])
    fpredictor.training_label = np.concatenate([predictor1.training_label, predictor2.training_label])

    fpredictor.validation_data = np.concatenate([predictor1.validation_data, predictor2.validation_data])
    fpredictor.validation_label = np.concatenate([predictor1.validation_label, predictor2.validation_label])

    fpredictor.all_data = np.concatenate([predictor1.all_data, predictor2.all_data])
    fpredictor.all_label = np.concatenate([predictor1.all_label, predictor2.all_label])

    print("Predictor1 Data")
    print(pd.DataFrame(predictor1.all_label).value_counts())
    print(pd.DataFrame(predictor1.training_label).value_counts())
    print(pd.DataFrame(predictor1.validation_label).value_counts())

    print("predictor2 Data")
    print(pd.DataFrame(predictor2.all_label).value_counts())
    print(pd.DataFrame(predictor2.training_label).value_counts())
    print(pd.DataFrame(predictor2.validation_label).value_counts())

    print("fpredictor Data")
    print(pd.DataFrame(fpredictor.all_label).value_counts())
    print(pd.DataFrame(fpredictor.training_label).value_counts())
    print(pd.DataFrame(fpredictor.validation_label).value_counts())

    fpredictor.create_model()
    fpredictor.epochs = 30
    fpredictor.train_model(all_data_bool=True)

    print('### Training Metrics ###')
    predictions = fpredictor.predict_model(fpredictor.training_data)
    final_preds = fpredictor.onehot_to_label(predictions)
    voted_preds = fpredictor.implement_voting(final_preds)
    voted_labels = fpredictor.implement_voting(fpredictor.training_label)
    print("Without Voting")
    fpredictor.metrics(fpredictor.training_label, final_preds)
    print("With Voting")
    fpredictor.metrics(voted_labels, voted_preds)
    # for i in range(len(voted_labels)):
    #     if voted_labels[i] != voted_preds[i]:
    #         plot_final_data(fpredictor.training_data[i*20], voted_labels[i] + '_' + voted_preds[i],
    #                         'Data/wrong_training/', i*20)

    print('### Validation Metrics ###')
    predictions = fpredictor.predict_model(fpredictor.validation_data)
    final_preds = fpredictor.onehot_to_label(predictions)
    voted_preds = fpredictor.implement_voting(final_preds)
    voted_labels = fpredictor.implement_voting(fpredictor.validation_label)
    print("Without Voting")
    fpredictor.metrics(fpredictor.validation_label, final_preds)
    print("With Voting")
    fpredictor.metrics(voted_labels, voted_preds)
    # for i in range(len(voted_labels)):
    #     if voted_labels[i] != voted_preds[i]:
    #         plot_final_data(fpredictor.validation_data[i*20], voted_labels[i] + '_' + voted_preds[i],
    #                         'Data/wrong_validation/', i*20)

    print('done')
