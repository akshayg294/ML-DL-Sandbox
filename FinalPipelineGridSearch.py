import copy
from FinalPipelinFunctions import *


if __name__ == '__main__':
    print('Starting Main')
    predictor1 = pickle.load(open('Data/predictor_180821_L25.pkl', 'rb'))
    predictor2 = pickle.load(open('Data/predictor_110821_L25.pkl', 'rb'))
    print('Read Data')
    fpredictor = Predictor()
    fpredictor.label_finallist = predictor1.label_finallist + predictor2.label_finallist
    fpredictor.nolabel_finallist = predictor1.nolabel_finallist + predictor2.nolabel_finallist

    lookbackrange = [30, 40, 50, 60]
    windowrange = [20, 25, 30]
    weight_vector = [False, True]
    dropoutrange = [0]
    learningraterange = [.0005, .001, .0015]
    window_threshold_range = [.25, .5, .75]
    batch_size_range = [512, 256, 128]
    num_of_experiments = 1
    resumer = 203  # Change this value to 1 + the max value in savedir
    saveind = 0
    savedir = 'Data/GridSearch07092021/'
    print('Entering main loop')
    for lookback in tqdm(lookbackrange, 'lookback'):
        for window in tqdm(windowrange, 'window'):
            predictor = copy.deepcopy(fpredictor)
            predictor.lookback = lookback
            predictor.window = window
            predictor.label_train, predictor.label_targets = predictor.create_dataset()
            predictor.nolabel_train, predictor.nolabel_targets = predictor.create_dataset(labelled_bool=False)
            predictor.all_train_test_dataset()
            for weight in tqdm(weight_vector, 'weight'):
                predictor.create_model(useweights=weight)
                for dropout in tqdm(dropoutrange, 'dropout'):
                    for learningrate in tqdm(learningraterange, 'learningrate'):
                        for batch_size in tqdm(batch_size_range, 'batchsize'):
                            if saveind >= resumer:
                                val_voted_preds = list()
                                val_voted_labels = list()
                                tra_voted_preds = list()
                                tra_voted_labels = list()
                                for j in tqdm(range(num_of_experiments), 'num_of_experiments'):
                                    print(saveind)
                                    trpredictor = copy.deepcopy(predictor)
                                    trpredictor.epochs = 30
                                    trpredictor.dropout = dropout
                                    trpredictor.learning_rate = learningrate
                                    trpredictor.batchsize = batch_size
                                    trpredictor.train_model(all_data_bool=False)

                                    # print('### Training Metrics ###')
                                    predictions = trpredictor.predict_model(trpredictor.training_data)
                                    final_preds = trpredictor.onehot_to_label(predictions)
                                    tra_voted_preds.append(final_preds)
                                    tra_voted_labels.append(trpredictor.training_label)

                                    # print('### Validation Metrics ###')
                                    predictions = trpredictor.predict_model(trpredictor.validation_data)
                                    final_preds = trpredictor.onehot_to_label(predictions)
                                    val_voted_preds.append(final_preds)
                                    val_voted_labels.append(trpredictor.validation_label)
                                    del trpredictor
                                save_values = ExperimentTracker()
                                save_values.lookback = lookback
                                save_values.window = window
                                save_values.learningrate = learningrate
                                save_values.dropout = dropout
                                save_values.weight = weight
                                save_values.batch_size = batch_size
                                save_values.val_voted_labels = val_voted_labels
                                save_values.val_voted_preds = val_voted_preds
                                save_values.tra_voted_preds = tra_voted_preds
                                save_values.tra_voted_labels = tra_voted_labels
                                with open(savedir + str(saveind) + '.pkl', 'wb') as file:
                                    pickle.dump(save_values, file)
                                saveind += 1
                            else:
                                saveind += 1
