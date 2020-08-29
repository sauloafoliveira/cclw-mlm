from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np

import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# from mlxtend.data import iris_data
# from mlxtend.plotting import plot_decision_regions

from mlm import MinimalLearningMachineClassifier as MLMC
from mlm import ClassCornerLightWeightedMinimalLearningMachineClassifier as CCLWMLM
from mlm import RandomMinimalLearningMachine as RMLM
from mlm import RankMinimalLearningMachineClassifier as RankMLM

from local_datasets import load_banana, load_vertebral_column_2c

bunch = load_vertebral_column_2c()
X, y = bunch.data, bunch.target


rmlm = RankMLM()
rmlm.fit(X, y)

print(rmlm.score(X, y))