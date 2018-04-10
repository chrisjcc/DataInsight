# ---- Import common python libraries
from __future__ import print_function
import sys
import time
import numpy as np
import pandas as pd
import random
import collections

# ---- Import from root_numpy library 
import root_numpy
from root_numpy import root2array, rec2array

# ---- Import from root_pandas library
import root_pandas
from root_pandas import read_root

# ---- Import from matplotlib
import matplotlib.pyplot as plt

# ---- Import Keras wrapper
from keras.wrappers.scikit_learn import KerasClassifier

# ----- Import scikit-learn
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

# ---- Import Local Interpretable Model-agnostic Explanations (LIME)
import lime
import lime.lime_tabular

# ---- Fix random seed for reproducibility
seed = 42
np.random.seed(seed)

# ---- Feature names
feature_names = [
    'n_jet', 
    'n_bjet', 'HT', 'MET',
    'METPhi', 
    'lep_type', 
    'lep_charge', 'MT', 'MT2W', 'M_lb', 'C', 'Y', 
    'HT_ratio', 'R_lb', 'Phi_jl', 'Phi_Wl',
]

# --- Load dataset
signal_sample     = 'training_sample_new/stop_sample.root'
background_sample = 'training_sample_new/top_sample.root'
tree_name         = 'outtree'

dataloader = DataLoader(signal_sample, background_sample, tree_name, feature_names)
data = dataloader.data
print("Total number of events: {}\nNumber of features: {}\n".format(data.shape[0], data.shape[1]))

# ---- Create features dataframe and target array
X = data.drop(["y"], axis=1, inplace=False)
y = data["y"]
input_dim = X.shape[1]

# ---- Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

# ---- Preprocessing using 0-1 scaling by removing the mean and scaling to unit variance
scaler = RobustScaler()

# ---- Plotting utility
plotter = Plotter()

# ---- Set deep neural network architecture: [m input] -> [n neurons] -> [1 output]
dnn = DeepModel(input_dim=input_dim)

# ---- Store network architecture
#plot_model(dnn.build_dnn_fn(), to_file='plots/model_dnn.pdf', show_shapes=True, show_layer_names=True)

# ---- Create model for use in scikit-learn
pipe_classifiers = {
    'kerasclassifier':  make_pipeline(scaler,  KerasClassifier(build_fn=dnn.build_dnn_fn, batch_size=256, epochs=80, verbose=2))
    }

# ---- Model fitting
pipe_classifiers['kerasclassifier'].fit(X_train, y_train)


## Model prediction interpretability with LIME

# ---- Use LIME to explain individual predictions, initialize explainer object
explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train.values,  # training data
                                                   mode='classification',
                                                   feature_names=feature_names,   # names of all features (regardless of type)
                                                   class_names=[0, 1],            # class names
                                                   discretize_continuous=True,
                                                   categorical_features= None,
                                                   categorical_names=None,
                                                   kernel_width=3
                                                   )


# ---- Establish predictor
estimator = pipe_classifiers['kerasclassifier']
predict_fn_kerasclassifier=lambda x: estimator.predict_proba(x).astype(float)


# ---- Explain a prediction ("local interpretability"): 
exp = explainer.explain_instance(data_row=X_test.values[0],   # 2d numpy array, corresponding to a row
                                 predict_fn=predict_fn_kerasclassifier,  # classifier prediction probability function, 
                                 labels=[0, 1],               # iterable with labels to be explained.
                                 num_features=input_dim,      # maximum number of features present in explanation
                                 top_labels=0,                # explanations for the K labels with highest prediction probabilities,
                                 num_samples=2000,            # size of the neighborhood to learn the linear model
                                 distance_metric='euclidean'  # the distance metric to use for weights.
                                 )


# ---- list values
exp.as_list()

# ---- Show explanation
exp.show_in_notebook(show_table=True, 
                     show_all=False, #True, 
                     )

fig = exp.as_pyplot_figure()
exp.save_to_file('lime_explanation_oi.html')

print(exp.available_labels())

# ---- Background class
print ('\n'.join(map(str, exp.as_list(label=0))))

# ---- Signal class
print ('\n'.join(map(str, exp.as_list(label=1))))

# ---- Local explanation for class 1 and 0
exp.as_pyplot_figure()

print('Couples probability:', exp.predict_proba[0])
