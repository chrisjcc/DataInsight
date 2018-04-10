# ---- Import common python libraries
from __future__ import print_function
import sys
import time
import numpy as np
import pandas as pd
import random
import collections
from time import time

# ---- Import from root_numpy library 
import root_numpy
from root_numpy import root2array, rec2array

# ---- Import from root_pandas library
import root_pandas
from root_pandas import read_root

# ---- Import from matplotlib
import matplotlib.pyplot as plt

# ----- Import scikit-learn
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.feature_selection import mutual_info_classif

# ---- Keras deep neural network library
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l1, l2 #,WeightRegularizer
from keras.constraints import maxnorm
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping

# ---- Scikit-Learn Optimizer
from skopt import gp_minimize, forest_minimize
from skopt.plots import plot_convergence
from skopt.plots import plot_evaluations

# ---- Import data loader
from neural_network.dataloaders import DataLoader

# ---- Import neural network modelling
from neural_network.models import DeepModel

# ---- Import optimization for hyper-parameter
from neural_network.optimization import SkOptObjective

# ---- Import plotter
from visualization.plotter import Plotter

# ---- Fix random seed for reproducibility
seed = 42
np.random.seed(seed)


# ---- Feature names
features = [
    "mass_tag_tag_min_deltaR", "median_mass_jet_jet",
    "maxDeltaEta_tag_tag", "mass_higgsLikeDijet", "HT_tags",
    "btagDiscriminatorAverage_tagged", "mass_jet_tag_min_deltaR",
    "mass_jet_jet_min_deltaR", "mass_tag_tag_max_mass", "maxDeltaEta_jet_jet",
    "centrality_jets_leps", "centrality_tags"
]

# --- Load dataset
signal_sample     = "data/signalMC.root"
background_sample = "data/backgroundMC.root"
tree_category     = "event_mvaVariables_step7_cate4"

dataloader = DataLoader(signal_sample, background_sample, tree_category, features)
data = dataloader.data
print("Total number of events: {}\nNumber of features: {}\n".format(data.shape[0], data.shape[1]))

# ---- Create features dataframe and target array
X = data.drop(["y"], axis=1, inplace=False)
y = data["y"]
input_dim = X.shape[1]
verbose = 2

# ---- Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

# ---- Preprocessing using 0-1 scaling by removing the mean and scaling to unit variance
scaler = RobustScaler()

# ---- Set deep neural network architecture: [m input] -> [n neurons] -> [1 output]
dnn = DeepModel(input_dim=input_dim)
dnn_model = dnn.build_dnn_fn(input_dim=input_dim)

# ---- Use early stopping on training when the validation loss isn't decreasing anymore
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# ---- Store network architecture
plot_model(dnn_model, to_file='plots/model_build_dnn_fn.pdf', show_shapes=True, show_layer_names=True)

# ---- Fit model on training data (use 10% of data for model validation)
# This `fit` call will be distributed on 4 GPUs.
# Since the batch size is 256, each GPU will process 64 samples.
start = time()
dnn_model_history = dnn_model.fit(X_train, y_train, validation_split=0.20,
                                  batch_size=256, epochs=80, verbose=verbose,
                                  callbacks=[early_stopping]
)
print("Model training took %.2f seconds." % (time() - start))

# ---- Evaluate model test performance
loss, accuracy =  dnn_model.evaluate(X_test, y_test, verbose=verbose)
print('\nEvaluate test loss \n%s: %.2f%%'    % (dnn_model.metrics_names[0], loss*100))
print('Evaluate test accuracy \n%s: %.2f%%'  % (dnn_model.metrics_names[1], accuracy*100))

# ---- Evaluate model training performance 
loss, accuracy = dnn_model.evaluate(X_train, y_train, verbose=verbose)
print('Evaluate train loss \n%s: %.2f%%'     % (dnn_model.metrics_names[0], loss*100))
print('Evaluate train accuracy \n%s: %.2f%%' % (dnn_model.metrics_names[1], accuracy*100))

# ---- List all data in history
print(dnn_model_history.history.keys())

## Visualize Model Training History
# ---- Summarize history for accuracy
plt.plot(dnn_model_history.history['acc'])
plt.plot(dnn_model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# ---- Summarize history for loss
plt.plot(dnn_model_history.history['loss'])
plt.plot(dnn_model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# ---- Calculate AUC of ROC
predictions = dnn_model.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, predictions)
roc_auc = auc(fpr, tpr)

# Plot all ROC curves
plt.plot(fpr, tpr, lw=2, label='%s (AUC = %0.3f)'%('DNN', roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("Receiver operating characteristic curve")
leg = plt.legend(loc="best", frameon=True, fancybox=True, fontsize=8)
leg.get_frame().set_edgecolor('w')
frame = leg.get_frame()
frame.set_facecolor('White')
print('AUC: %f' % roc_auc)
plt.show()


# Estimate mutual information (MI) for a discrete target variable. MI measures the dependency between the variables 
# relies on nonparametric methods based on entropy estimation from k-nearest neighbors distances
mi = mutual_info_classif(X_train, y_train)
inds = np.argsort(mi)

plt.figure(figsize=(5, 30))
plt.barh(np.arange(len(mi)), np.log(mi[inds] + 1))
plt.yticks(np.arange(len(mi)), X_train.columns[inds])
plt.ylim(0, len(mi))
plt.show()
