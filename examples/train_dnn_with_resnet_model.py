# ---- Import common python libraries
from __future__ import print_function
import sys
from time import time
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

# ----- Import scikit-learn
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score

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
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.utils import plot_model, np_utils

# ---- Import data loader
from neural_network.dataloaders import DataLoader

# ---- Import neural network modelling
from neural_network.models import DeepModel
from neural_network.models import ResnetBuilder

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
    "centrality_jets_leps", "centrality_tags", "globalTimesEventWeight"
]

# --- Load dataset
signal_sample     = "data/signal.root"
background_sample = "data/background.root"
treename          = "event_mvaVariables_step7_cate4"

dataloader = DataLoader(signal_sample, background_sample, treename, features)
data = dataloader.data
print("Total number of events: {}\nNumber of features: {}\n".format(data.shape[0], data.shape[1]))

# ---- Create features dataframe and target array
X = data.drop(["y","globalTimesEventWeight"], axis=1, inplace=False)
y = data["y"]
input_dim = X.shape[1]+1 # add 1 to foresee the use of ResNet score as input to DNN

# ---- A standard split of the dataset is used to evaluate and compare models
x_train, x_test, train_y, test_y = train_test_split(X, y, test_size=0.10, random_state=42)

# ---- Preprocessing using 0-1 scaling by removing the mean and scaling to unit variance
scaler = RobustScaler()

# ---- Plotting utility
plotter = Plotter()

# ---- Set deep neural network architecture: [m input] -> [n neurons] -> [1 output]
dnn = DeepModel(input_dim=input_dim)
dnn_model = dnn.build_dnn_fn()

# ---- Store network architecture
#plot_model(dnn.build_dnn_fn(), to_file='plots/model_dnn.pdf', show_shapes=True, show_layer_names=True)


features          = [
    'lepton_1_px', 'lepton_1_py', 'lepton_1_pz',
    'lepton_2_px', 'lepton_2_py', 'lepton_2_pz',
    'jet_1_px', 'jet_1_py', 'jet_1_pz', #'jet_1_btag_discrim', #'jet_1_pe',
    'jet_2_px', 'jet_2_py', 'jet_2_pz', #'jet_2_btag_discrim', #'jet_2_pe',
    'jet_3_px', 'jet_3_py', 'jet_3_pz', #'jet_3_btag_discrim', #'jet_3_pe',
    'jet_4_px', 'jet_4_py', 'jet_4_pz', #'jet_4_btag_discrim', #'jet_4_pe',
    'jet_5_px', 'jet_5_py', 'jet_5_pz', #'jet_5_btag_discrim', #'jet_5_pe',
    'jet_6_px', 'jet_6_py', 'jet_6_pz', #'jet_6_btag_discrim'  #'jet_6_pe',
    ]

verbose = 2

# ---- Input image dimensions (use 8 x 8 pixel image)
img_channels = 2
img_depth, img_width, img_height = img_channels, 8, 8 #img_rows, img_cols

# ---- Set fitting parameters
batch_size = 256
nb_classes = 2
epochs     = 80
data_augmentation = True

# ---- Image input data is expressed as a 3-dim matrix of channels x width x height
input_shape = (img_depth, img_width, img_height)

# ---- Load the analysis dataset
dataloader =  DataLoader(signal_sample, background_sample, treename, features)
print("Total number of events: {}\nNumber of features (incl. class label): {}\n".format(dataloader.data.shape[0],
                                                                                        dataloader.data.shape[1]))

# ---- Scikit-learn style dataset format from pandas dataframe (example here is sampling from dataset)
frac=1. #1e-2
sampled_data = dataloader.data#.sample(frac=frac, replace=False, random_state=42)

X = sampled_data.drop(['y'], axis=1, inplace=False)
y = sampled_data['y']

# ---- A standard split of the dataset is used to evaluate and compare models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# ---- Create two-dim image training & test dataset
start = time()
X_train_jet_img = dataloader.create_image(X_train, 'jet', input_shape)
print("Took %.2f seconds to transform training data into image data format."  % (time() - start))
X_test_jet_img  = dataloader.create_image(X_test,  'jet', input_shape)

start = time()
X_train_lepton_img = dataloader.create_image(X_train, 'lepton', input_shape)
print("Took %.2f seconds to transform training data into image data format."  % (time() - start))
X_test_lepton_img  = dataloader.create_image(X_test, 'lepton', input_shape)

X_train_jet_img    = X_train_jet_img.reshape(X_train_jet_img.shape[0], X_train_jet_img.shape[2], X_train_jet_img.shape[3], X_train_jet_img.shape[1])
X_test_jet_img     = X_test_jet_img.reshape(X_test_jet_img.shape[0], X_test_jet_img.shape[2], X_test_jet_img.shape[3], X_test_jet_img.shape[1])
X_train_lepton_img = X_train_lepton_img.reshape(X_train_lepton_img.shape[0], X_train_lepton_img.shape[2], X_train_lepton_img.shape[3], X_train_lepton_img.shape[1])
X_test_lepton_img  = X_test_lepton_img.reshape(X_test_lepton_img.shape[0], X_test_lepton_img.shape[2], X_test_lepton_img.shape[3], X_test_lepton_img.shape[1])

# ---- Concatenate the various imagine depth (use axis=1 since reshape is already applied)
Images_array_train = np.concatenate((X_train_jet_img, X_train_lepton_img), axis=3)
Images_array_test  = np.concatenate((X_test_jet_img,  X_test_lepton_img),  axis=3)

# ---- Convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

# ---- Set model training monitoring options
lr_reducer    = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger    = CSVLogger('resnet34_logger.csv')

# ---- Build Residual network model (type 34)
resnet = ResnetBuilder()
resnet_model = resnet.build_resnet_34((img_channels, img_height, img_width), nb_classes)
resnet_model.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])

if not data_augmentation:
    print('Not using data augmentation.')
    resnet_model_history = resnet_model.fit(Images_array_train, y_train,
                                            batch_size=batch_size,
                                            epochs=epochs,
                                            validation_data=(Images_array_test, y_test),
                                            shuffle=True,
                                            callbacks=[lr_reducer,
                                                       csv_logger,
                                                       #early_stopper
                                                   ])
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,             # set input mean to 0 over the dataset
        samplewise_center=False,              # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,   # divide each input by its std
        zca_whitening=False,                  # apply ZCA whitening
        rotation_range=0,                     # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,                # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,               # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,                 # randomly flip images
        vertical_flip=False)                  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(Images_array_train)

    # Fit the model on the batches generated by datagen.flow().
    start = time()
    resnet_model_history = resnet_model.fit_generator(datagen.flow(Images_array_train, y_train, batch_size=batch_size),
                                                      steps_per_epoch=Images_array_train.shape[0] // batch_size,
                                                      validation_data=(Images_array_test, y_test),
                                                      epochs=epochs, verbose=1, max_q_size=100,
                                                      callbacks=[lr_reducer, early_stopper, csv_logger])
    print("Model training took %.2f seconds." % (time() - start))

# ---- Evaluate model test performance
loss, accuracy = resnet_model.evaluate(Images_array_test, y_test, verbose=verbose)
print('\nEvaluate test loss \n%s: %.2f%%'    % (resnet_model.metrics_names[0], loss*100))
print('Evaluate test accuracy \n%s: %.2f%%'  % (resnet_model.metrics_names[1], accuracy*100))

# ---- Evaluate model training performance 
loss, accuracy = resnet_model.evaluate(Images_array_train, y_train, verbose=verbose)
print('Evaluate train loss \n%s: %.2f%%'     % (resnet_model.metrics_names[0], loss*100))
print('Evaluate train accuracy \n%s: %.2f%%' % (resnet_model.metrics_names[1], accuracy*100))

# ---- Store network achitecture diagram into png
plot_model(resnet_model, to_file='plots/model_resnet.pdf')

# ---- List all data in history
print(resnet_model_history.history.keys())

# ---- Calculate AUC of ROC
resnet_scores = resnet_model.predict(Images_array_train)
resnet_scores = [item[1] for item in resnet_scores]
x_train["ResNet_score"] = resnet_scores

history = dnn_model.fit(x_train, train_y)

resnet_scores = resnet_model.predict(Images_array_test)
resnet_scores = [item[1] for item in resnet_scores]
x_test["ResNet_score"] = resnet_scores

predictions = dnn_model.predict(x_test)

fpr, tpr, _ = roc_curve(test_y, predictions)
roc_auc = auc(fpr, tpr)

# Plot all ROC curves
plt.plot(fpr, tpr, lw=2, label='%s (AUC = %0.3f)'%('ResNet version', roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("Receiver operating characteristic curve (%s events)" % (len(y_test)))
leg = plt.legend(loc="best", frameon=True, fancybox=True, fontsize=8)
leg.get_frame().set_edgecolor('w')
frame = leg.get_frame()
frame.set_facecolor('White')
print('AUC: %f' % roc_auc)
plt.show()
