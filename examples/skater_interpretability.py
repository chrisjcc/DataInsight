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
from keras.metrics import sparse_categorical_accuracy

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
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.20, random_state=seed)

# ---- Preprocessing using 0-1 scaling by removing the mean and scaling to unit variance
scaler = RobustScaler()

# ---- Use early stopping on training when the validation loss isn't decreasing anymore
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Define Multilayer Perceptron architecture
def create_model(input_dim=7342, n_classes = 11, nlayers=5, nneurons=100,
                 dropout_rate=0.0, l2_norm=0.001, learning_rate=1e-3,
                 activation='relu', kernel_initializer='lecun_normal',
                 optimizer='adam', metric=sparse_categorical_accuracy, 
                 loss='sparse_categorical_crossentropy'):
    '''
    create_model
    '''
        
    # create neural network model
    model = Sequential()
    
    # Add fully connected layer with an activation function (input layer)
    model.add(Dense(units=nneurons,
                    input_dim=input_dim,
                    kernel_initializer=kernel_initializer,
                    activation=activation,
                    kernel_regularizer=l2(l2_norm)))
    
    if dropout_rate != 0.:
        model.add(Dropout(dropout_rate))
                                        
    # Indicate the number of hidden layers
    for index, layer in enumerate(range(nlayers-1)):
        model.add(Dense(units=nneurons,
                        kernel_initializer=kernel_initializer,
                        activation=activation,
                        kernel_regularizer=l2(l2_norm)))
        
    # Add dropout layer
    if dropout_rate != 0.:
        model.add(Dropout(dropout_rate))
        
    # Add fully connected output layer with a sigmoid activation function
    model.add(Dense(n_classes,#1, #n_classes,
                    kernel_initializer=kernel_initializer,
                    activation='softmax',
                    kernel_regularizer=l2(l2_norm)))
    
    # Compile neural network (set loss and optimize)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=[metric]) #'crossentropy'
    
    # Print summary report
    if True:
        model.summary()
    
    # Return compiled network
    return model


from skater.core.explanations import Interpretation
from skater.model import InMemoryModel

# Configure neural network architecture
input_dim    = X.values.shape[1]
n_classes    = len(list(set(y)))
nlayers      = 3  
nneurons     = 51
l2_norm      = 0.0014677547170664112
dropout_rate = 0.014655354118727714
loss         = 'sparse_categorical_crossentropy'

default_parameters = [3, 51, 0.0014677547170664112, 0.014655354118727714]
print('input_dim', input_dim)
print('n_classes', n_classes)

# create model for use in scikit-learn
binary_pipe = {
    'kerasclassifier':  make_pipeline(scaler,
                                      KerasClassifier(build_fn=create_model,
                                                      input_dim=input_dim,
                                                      n_classes=n_classes,
                                                      nlayers=nlayers,
                                                      nneurons=nneurons,
                                                      dropout_rate=dropout_rate,
                                                      l2_norm=l2_norm,
                                                      loss=loss,
                                                      batch_size=256, 
                                                      epochs=35,
                                                      verbose=1))
}

## Applying Model Agnostic Interpretation to Ensemble Models
# source:
#   - https://github.com/datascienceinc/Skater/blob/master/examples/ensemble_model.ipynb

interpreter = Interpretation(X_test, feature_names=features)

estimator = binary_pipe['kerasclassifier']
estimator.fit(X_train, y_train)

model = InMemoryModel(estimator.predict_proba, 
                      examples=X_train)

plots = interpreter.feature_importance.plot_feature_importance(model, ascending=True)

# Use partial dependence to understand the relationship between a variable and a model's predictions
model = InMemoryModel(estimator.predict_proba,
                      examples=X_test,
                      #unique_values=model.classes_
                      target_names=list(set(y_train)))

# Lets understand interaction using 2-way interaction using the same covariates                                                     
# feature_selection
p = interpreter.partial_dependence.plot_partial_dependence(features,
                                                           model, 
                                                           grid_resolution=50, #30 
                                                           with_variance=True,
                                                           figsize=(10, 5))
        
model_feature_interaction = InMemoryModel(estimator.predict_proba, examples=X_train, target_names=['acdc', 
                                                                                  'non_acdc'])
interpreter.partial_dependence.plot_partial_dependence([("maxDeltaEta_tag_tag", "mass_higgsLikeDijet")], 
                                                       model, 
                                                       grid_resolution=10)


from skater.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer

exp = LimeTabularExplainer(X_train, 
                           feature_names=features,
                           class_names=['acdc', 'non_acdc'],
                           discretize_continuous=True)
plt.show()

# explain prediction for data point for background label 'non_acdc'
exp.explain_instance(X_test[1], estimator.predict_proba)


# Interactive slider for controlling grid resolution
def understanding_interaction():
    pyint_model = InMemoryModel(estimator.predict_proba, examples=X_test, target_names=features)
    # ['worst area', 'mean perimeter'] --> list(feature_selection.value)
    interpreter.partial_dependence.plot_partial_dependence(["mass_tag_tag_max_mass", "maxDeltaEta_jet_jet"],
                                                           model, 
                                                           grid_resolution=grid_resolution.value,
                                                           with_variance=True)
        
    # Lets understand interaction using 2-way interaction using the same covariates
    # feature_selection.value --> ('worst area', 'mean perimeter')
    axes_list = interpreter.partial_dependence.plot_partial_dependence(["mass_tag_tag_max_mass", "maxDeltaEta_jet_jet"],
                                                                       pyint_model, 
                                                                       grid_resolution=grid_resolution.value,
                                                                       with_variance=True)


# One could further improve this by setting up an event callback using
# asynchronous widgets
import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import display
from IPython.display import clear_output
grid_resolution = widgets.IntSlider(description="GR", 
                                    value=10, min=10, max=100)
display(grid_resolution)

# dropdown to select relevant features from the dataset
feature_selection = widgets.SelectMultiple(
    options=tuple(features),
    value=["mass_tag_tag_max_mass", "maxDeltaEta_jet_jet"],
    description='Features',
    layout=widgets.Layout(display="flex", flex_flow='column', align_items = 'stretch'),
    disabled=False,
    multiple=True
)
print(feature_selection)

# Reference: http://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Events.html
button = widgets.Button(description="Generate Interactions")
display(button)

def on_button_clicked(button_func_ref):
    clear_output()
    understanding_interaction()

button.on_click(on_button_clicked)


from skater.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer
from IPython.display import display, HTML, clear_output
int_range = widgets.IntSlider(description="Index Selector", value=9, min=0, max=100)
display(int_range)

def on_value_change(change):
    index = change['new']
    exp = LimeTabularExplainer(X_test, 
                               feature_names=features, 
                               discretize_continuous=False, 
                               class_names=['acdc', 'non_acdc'])
    print("Model behavior at row: {}".format(index))
    # Lets evaluate the prediction from the model and actual target label
    print("prediction from the model:{}".format(estimator.predict(X_test[index].reshape(1, -1))))
    print("Target Label on the row: {}".format(y_test.reshape(1,-1)[0][index]))
    clear_output()
    display(HTML(exp.explain_instance(X_test[index], models['ensemble'].predict_proba).as_html()))

int_range.observe(on_value_change, names='value')
