## Create a Convolutional Neural Network with Keras
from __future__ import print_function

# ---- Import basic libraries
import matplotlib.pyplot as plt
import numpy as np
from time import time

# ---- Import Keras library
import keras
from keras.models import model_from_json
from keras.utils import plot_model
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Input

# ---- Import Scikit-learn library
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score

# ---- Import data loader
from neural_network.dataloaders import DataLoader

# ---- Import neural network modelling
from neural_network.models import DeepModel

# ---- Import plotter
from visualization.plotter import Plotter


# ---- Declare sample and features to use
signal_sample     = 'data/signal.root'
background_sample = 'data/background.root'
treename          = 'event_mvaVariables_step7_cate4'

features          = [
    'lepton_1_px', 'lepton_1_py', 'lepton_1_pz',
    'lepton_2_px', 'lepton_2_py', 'lepton_2_pz',
    'jet_1_px', 'jet_1_py', 'jet_1_pz',
    'jet_2_px', 'jet_2_py', 'jet_2_pz',
    'jet_3_px', 'jet_3_py', 'jet_3_pz',
    'jet_4_px', 'jet_4_py', 'jet_4_pz',
    'jet_5_px', 'jet_5_py', 'jet_5_pz',
    'jet_6_px', 'jet_6_py', 'jet_6_pz',
    ]

verbose = 2

# ---- Dimensions of our images (use 8 x 8 pixel image)
channel = 2
img_depth, img_width, img_height = channel, 64, 64

# ---- Image input data is expressed as a 3-dim matrix of channels x width x height
input_shape = (img_depth, img_width, img_height)

# ---- Load the analysis dataset
dataloader =  DataLoader(signal_sample, background_sample, treename, features)
print("Total number of events: {}\nNumber of features (incl. class label): {}\n".format(dataloader.data.shape[0],
                                                                                        dataloader.data.shape[1]))

# ---- Scikit-learn style dataset format from pandas dataframe (example here is sampling from dataset)
frac=1e-2
sampled_data = dataloader.data.sample(frac=frac, replace=False, random_state=42)

X = sampled_data.drop(['y'], axis=1, inplace=False)
y = sampled_data['y']

# ---- A standard split of the dataset is used to evaluate and compare models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# ---- Create two-dim image training & test dataset
start = time()
X_train_jet_img = dataloader.create_image(X_train, 'jet', input_shape)
print("Took %.2f seconds to transform training data into image data format."  % (time() - start))
X_test_jet_img  = dataloader.create_image(X_test,  'jet', input_shape)

start = time()
X_train_lepton_img = dataloader.create_image(X_train, 'lepton', input_shape)
print("Took %.2f seconds to transform training data into image data format."  % (time() - start))
X_test_lepton_img  = dataloader.create_image(X_test, 'lepton', input_shape)

# ---- Concatenate the various imagine depth (use axis=1 since reshape is already applied)
Images_array_train = np.concatenate((X_train_jet_img, X_train_lepton_img), axis=1)
Images_array_test  = np.concatenate((X_test_jet_img,  X_test_lepton_img),  axis=1)


## Convolutional Neural Network Modeling (CNN)
# ---- Define CNN network architecture modeling
cnn       = DeepModel(input_shape=input_shape)
cnn_model = cnn.build_cnn_fn(input_shape=input_shape)

# ---- Store network achitecture diagram into png
plot_model(cnn_model, to_file='plots/model_build_cnn_fn.png')

# ---- Use early stopping on training when the validation loss isn't decreasing anymore
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# ---- Fit model on training data (use 10% of data for model validation)
start = time()
cnn_model_history = cnn_model.fit(Images_array_train, y_train, validation_split=0.20,
                                  batch_size=256, epochs=80, verbose=verbose,
                                  callbacks=[early_stopping]
                                  )
print("Model training took %.2f seconds." % (time() - start))

# ---- Evaluate model test performance
loss, accuracy = cnn_model.evaluate(Images_array_test, y_test, verbose=verbose)
print('\nEvaluate test loss \n%s: %.2f%%'    % (cnn_model.metrics_names[0], loss*100))
print('Evaluate test accuracy \n%s: %.2f%%'  % (cnn_model.metrics_names[1], accuracy*100))

# ---- Evaluate model training performance 
loss, accuracy = cnn_model.evaluate(Images_array_train, y_train, verbose=verbose)
print('Evaluate train loss \n%s: %.2f%%'     % (cnn_model.metrics_names[0], loss*100))
print('Evaluate train accuracy \n%s: %.2f%%' % (cnn_model.metrics_names[1], accuracy*100))

# ---- List all data in history
print(cnn_model_history.history.keys())

## Visualize Model Training History
# ---- Summarize history for accuracy
plt.plot(cnn_model_history.history['acc'])
plt.plot(cnn_model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# ---- Summarize history for loss
plt.plot(cnn_model_history.history['loss'])
plt.plot(cnn_model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# ---- Calculate AUC of ROC
predictions = cnn_model.predict(Images_array_test)
fpr, tpr, _ = roc_curve(y_test, predictions)
roc_auc = auc(fpr, tpr)

# Plot all ROC curves
plt.plot(fpr, tpr, lw=2, label='%s (AUC = %0.3f)'%('CNN', roc_auc))
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
