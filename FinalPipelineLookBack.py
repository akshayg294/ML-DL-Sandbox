import copy
from FinalPipelinFunctions import *

if __name__ == '__main__':
    predictor1 = pickle.load(open('Data/predictor_180821_L25.pkl', 'rb'))
    predictor2 = pickle.load(open('Data/predictor_110821_L25.pkl', 'rb'))

    fpredictor = Predictor()
    fpredictor.label_finallist = predictor1.label_finallist + predictor2.label_finallist
    fpredictor.nolabel_finallist = predictor1.nolabel_finallist + predictor2.nolabel_finallist

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

    lookbackrange = np.arange(20, 81, 5)
    samplesize = 10
    f1score_array = np.empty((len(lookbackrange), samplesize))
    for i in tqdm(range(len(lookbackrange))):
        for j in tqdm(range(samplesize)):
            predictor = copy.deepcopy(fpredictor)
            predictor.lookback = lookbackrange[i]
            predictor.label_train, predictor.label_targets = predictor.create_dataset()
            predictor.nolabel_train, predictor.nolabel_targets = predictor.create_dataset(labelled_bool=False)
            predictor.all_train_test_dataset()
            predictor.create_model()
            predictor.epochs = 30
            predictor.train_model(all_data_bool=True)
            predictions = predictor.predict_model(predictor.validation_data)
            final_preds = predictor.onehot_to_label(predictions)
            voted_preds = predictor.implement_voting(final_preds)
            voted_labels = predictor.implement_voting(predictor.validation_label)
            f1score_array[i, j] = f1_score(voted_labels, voted_preds, average='macro')
            del predictor

    np.save('/Data/lookback_varying.npy', f1score_array)
    data = np.load('Data/lookback_varying.npy')
    lookbackrange = np.arange(20, 81, 5)
    plot_candlestick(data, lookbackrange)

