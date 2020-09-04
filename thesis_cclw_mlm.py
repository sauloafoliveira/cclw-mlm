from sklearn import datasets
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

import numpy as np
from os.path import exists
import pickle
import pandas as pd

from scipy.stats import uniform, loguniform

import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# from mlxtend.data import iris_data
# from mlxtend.plotting import plot_decision_regions

from mlm import MinimalLearningMachineClassifier as FullMLM
from mlm import ClassCornerLightWeightedMinimalLearningMachineClassifier as CCLWMLM
from mlm import RandomMinimalLearningMachine as RandomMLM
from mlm import RankMinimalLearningMachineClassifier as RankMLM

from local_datasets import load_banana, load_vertebral_column_2c, load_ripley, \
        load_german_stalog, load_habermans_survivor, load_statlog_heart, \
        load_ionosphere, load_breast_cancer, load_two_moon, load_pima_indians

datasets = {
    'BAN' : load_banana(),
    'BCW' : load_breast_cancer(),
    'GER' : load_german_stalog(),
    'HEA' : load_statlog_heart(),
    'HAB' : load_habermans_survivor(),
    'ION' : load_ionosphere(),
    'PID' : load_pima_indians(),
    'RIP' : load_ripley(),
    'TMN' : load_two_moon(),
    'VCP' : load_vertebral_column_2c()
}

# hyperparameter optmization
hyperparameters = None

OPTIMUM_PARAMS_FILENAME = 'optimum_parameters.pkl'
EXPERIMENT_BUNCHES_FILENAME = 'experiment_bunches.pkl'

classifiers = [
    ('Random-MLM', RandomMLM(), {
        'factor' : uniform(0.05, 0.5)
    }),
    ('Rank-MLM', RankMLM(), {
        'C' : loguniform(1e-3, 1e2)
    }),
    ('Full-MLM', FullMLM(), None),
    ('CCLW-MLM', CCLWMLM(), None) 
]

trials = 30

experiment_bunches = None

if exists(EXPERIMENT_BUNCHES_FILENAME):
    with open(EXPERIMENT_BUNCHES_FILENAME, 'rb') as f:
        experiment_bunches = pickle.load(f)
        print('Bunches recovered.')

if experiment_bunches == None:
    print('Creating bunches from scratch.')
    experiment_bunches = dict()

    # save samples for experiments
    for dataset_name, bunch in datasets.items():
        X, y = bunch.data, bunch.target

        bunch_list = []

        for _ in range(30):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

            scaler = StandardScaler()
            scaler.fit(X_train)

            X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

            bunch_list.append((X_train, X_test, y_train, y_test))
        
        experiment_bunches[dataset_name] = bunch_list

    with open(EXPERIMENT_BUNCHES_FILENAME, 'wb') as f:
        pickle.dump(experiment_bunches, f)
        print('Bunches saved.')


optimum_parameters = None
if exists(OPTIMUM_PARAMS_FILENAME):
    with open(OPTIMUM_PARAMS_FILENAME, 'rb') as f:
        optimum_parameters = pickle.load(f)
        print('Optimum Hyperparameters recovered.')

if optimum_parameters == None:

    print('Performing hyperparameter optimization.')

    optimum_parameters = {classif: {name: None for name in datasets.keys()} \
        for classif, _, _ in classifiers}

    scaler = StandardScaler()

    for classif, base_estimator, params in classifiers:
        optimum_parameters[classif] = {name: None for name in datasets.keys()}

        if params != None:
            for dataset_name, bunch in datasets.items():
                X, y = bunch.data, bunch.target
                
                clf = RandomizedSearchCV(base_estimator, params, random_state=0)
                clf.fit(scaler.fit_transform(X), y)

                optimum_parameters[classif][dataset_name] = clf.best_params_
        
    with open(OPTIMUM_PARAMS_FILENAME, 'wb') as f:
        pickle.dump(optimum_parameters, f)
        print('Optimum Hyperparameters saved.')   

print(optimum_parameters)

### Colect metrics to evaluate models

cols = ['classifier_name', 'dataset_name', 'accuracy', 'std_accuracy', 'norm', 'std_norm']
black_box_results = pd.DataFrame(columns=cols)

for classif, base_estimator, params in classifiers:
    
    for dataset_name, bunches in experiment_bunches.items():

        # load best params if any
        if params != None and dataset_name in optimum_parameters[classif]:            
            best_params = optimum_parameters[classif][dataset_name]
            
            base_estimator = base_estimator.set_params(**best_params)

        metrics = []

        for X_train, X_test, y_train, y_test in bunches:

            base_estimator.fit(X_train, y_train)

            trial_result = base_estimator.score(X_test, y_test), *base_estimator.sparsity()
            
            metrics.append(trial_result)

        m = np.mean(metrics, axis=0)
        s = np.std(metrics, axis=0)

        print(black_box_results)
        
        t = pd.Series(data=[classif, dataset_name, m[0], s[0], m[-1], s[-2]], index=cols)

        black_box_results = black_box_results.append(t, ignore_index=True)


print(black_box_results)
black_box_results.to_csv('results_thesis.csv')

print('Experiment concluded.')