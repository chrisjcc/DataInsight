# ---- Basic import libraries
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import math
import six

# ---- Import Keras deep neural network library
from keras.models import Sequential
from keras.models import Model
from keras.regularizers import l1, l2
from keras.optimizers import Adam, Adadelta, SGD
from keras.layers.normalization import BatchNormalization
from keras.layers import (
    Input, Dense, Dropout, Activation, Flatten, merge, Merge, concatenate,
    GlobalAveragePooling2D, LeakyReLU, UpSampling2D, GaussianDropout, Reshape
)
from keras.layers.convolutional import (
    Conv2D, Convolution2D, MaxPooling2D, AveragePooling2D
)

from keras.layers.merge import Concatenate
from keras.layers.core import Lambda
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.merge import add
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K


# ---- Base model
class BaseModel(object):

    # Model base constructor
    def __init__(self):
        self.estimator = None

    # Model functor
    def __call__(self):
        pass

# ---- Neural network model
class DeepModel(BaseModel):
    '''
    Deep model
    '''
    # Deep model constructor intialize model architecture
    def __init__(self, nlayers=1, list_of_nneurons=[50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
                 dropout_rate=0.0, l2_norm=0.001, learning_rate=1e-3,
                 activation='relu', kernel_initializer='lecun_normal', optimizer='adam',
                 input_dim=None, input_shape=None, metric='accuracy', loss='binary_crossentropy',
                 save_as='model.h5', report_summary=True, network_architecture="cnn"):
        '''
        Constructor
        '''
        self.nlayers=nlayers
        self.list_of_nneurons=list_of_nneurons
        self.dropout_rate=dropout_rate
        self.l2_norm=l2_norm
        self.learning_rate=learning_rate
        self.activation=activation
        self.kernel_initializer=kernel_initializer
        self.optimizer=optimizer
        self.input_dim=input_dim
        self.input_shape=input_shape
        self.metric=metric
        self.loss=loss
        self.save_as=save_as
        self.report_summary=report_summary
        self.network_architecture = network_architecture
        self.model = None

    # Function to create model architecture
    def __call__(self, **kwargs):
        '''
        Functor
        '''
        if "nlayers" in kwargs:
            self.nlayers = int(kwargs["nlayers"])
        if "dropout_rate" in kwargs:
            self.dropout_rate=kwargs["dropout_rate"]
        if "l2_norm" in kwargs:
            self.l2_norm=kwargs["l2_norm"]
        if "learning_rate" in kwargs:
            self.learning_rate=kwargs["learning_rate"]
        if "activation" in kwargs:
            self.learning_rate=kwargs["activation"]
        if "kernel_initializer" in kwargs:
            self.kernel_initializer=kwargs["kernel_initializer"]
        if "optimizer" in kwargs:
            self.optimizer=kwargs["optimizer"]
        if "input_dim" in kwargs:
            self.input_dim=int(kwargs["input_dim"])
        if "metric" in kwargs:
            self.metric=kwargs["metric"]
        if "loss" in kwargs:
            self.loss=kwargs["loss"]
        if "save_as" in kwargs:
            self.save_as=kwargs["save_as"]
        if "report_summary" in kwargs:
            self.report_summary=kwargs["report_summary"]
        if "network_architecture" in kwargs:
            self.network_architecture=kwargs["network_architecture"]
        
        return self

    # Function to create model: [m input] -> [n neurons] -> [1 output]
    def create_model(self):
        '''
        Create model
        '''
        print('Build model...')
        return self.build_dnn_fn(input_dim=self.input_dim, nlayers=self.nlayers, list_of_nneurons=self.list_of_nneurons,
                                 dropout_rate=self.dropout_rate, l2_norm=self.l2_norm, learning_rate=self.learning_rate,
                                 activation=self.activation, kernel_initializer=self.kernel_initializer,
                                 optimizer=self.optimizer, metric=self.metric, loss=self.loss,
                                 save_as=self.save_as, report_summary=self.report_summary)

    # Function for constructing the traditional deep neural network
    def build_dnn_fn(self,
                     nlayers=5, list_of_nneurons=[50, 50, 50, 50, 50, 50, 50, 50, 50, 50], 
                     dropout_rate=0.0, l2_norm=0.001, learning_rate=1e-3,
                     activation='relu', kernel_initializer='lecun_normal', optimizer='adam',
                     input_dim=12, metric='accuracy', loss='binary_crossentropy',
                     save_as='model_dnn.h5', report_summary=True):
        '''
        build_fn
        '''
        print('Build model...')

        # prevent from giving float value
        self.nlayers   = int(nlayers)
        self.dropout_rate = dropout_rate
        self.l2_norm = l2_norm
        self.learning_rate = learning_rate
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.optimizer = Adam(lr=self.learning_rate) if optimizer == None else optimizer
        # Setting up the optimization of our weights
        #self.optimizer = = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True) if optimizer == None else optimizer
        self.metric = metric
        self.loss = loss
        self.save_as = save_as
        self.report_summary = report_summary

        # Start neural network
        self.model = Sequential()

        # Add fully connected layer with an activation function (input layer)
        self.model.add(Dense(units=list_of_nneurons[0], # 1st hideen unit can be set to input_dim
                             input_dim=self.input_dim,
                             kernel_initializer=self.kernel_initializer,
                             activation=self.activation,
                             kernel_regularizer=l2(self.l2_norm)))

        # Add dropout layer
        if self.dropout_rate != 0.:
            self.model.add(Dropout(self.dropout_rate))

        # Indicate the number of hidden layers
        for index, layer in enumerate(range(self.nlayers-1)):
            self.model.add(Dense(units=list_of_nneurons[index+1], 
                                 kernel_initializer=self.kernel_initializer,
                                 activation=self.activation,
                                 kernel_regularizer=l2(self.l2_norm)))
            # Add dropout layer
            if self.dropout_rate != 0.:
                self.model.add(Dropout(self.dropout_rate))

        # Add fully connected output layer with a sigmoid activation function
        self.model.add(Dense(1,
                             kernel_initializer=self.kernel_initializer,
                             activation='sigmoid',
                             kernel_regularizer=l2(self.l2_norm)))

        # Compile neural network (set loss and optimize)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=self.optimizer,
                           metrics=[self.metric])

        # Print summary report
        if self.report_summary:
            self.model.summary()

        # Store model to file
        self.model.save(self.save_as)

        # Return compiled network
        return self.model

    # Function for constructing the convolution neural network
    def build_cnn_fn(self, input_shape=(1, 8, 8),
                     kernel_size=(2,2), pool_size=(1, 1),
                     dropout_rate=0.1, activation='relu', 
                     optimizer=Adadelta(), batch_norm=False, metric='accuracy', 
                     loss='binary_crossentropy', report_summary=True,
                     save_as='model_build_cnn_fn.h5'): 
        '''
        build_cnn_fn: creates convolutional neural network architecture.
        '''
        print('Build model...')
        print(str(input_shape))
        print(str(self.input_shape))

        # Start convolutional neural network 
        self.cnn_model = Sequential()
        
        # Convolutional layers are comprised of filters and feature maps
        # - The filters are essentially the neurons of the layer. 
        # They have both weighted inputs and generate an output value like a neuron. 
        # The input size is a fixed square called a patch or a receptive field. 
        # If the convolutional layer is an input layer, then the input patch will be pixel values. 
        # If they are deeper in the network architecture, then the convolutional layer will take 
        # input from a feature map from the previous layer.
        # - The feature map is the output of one filter applied to the previous layer. 
        # A given filter is drawn across the entire previous layer, moved one pixel at a time. 
        # Each position results in an activation of the neuron and the output is collected in the feature map.
        self.cnn_model.add(Conv2D(filters=32, #128, 
                                  kernel_size=(4,4), 
                                  padding='same',
                                  activation=activation,
                                  input_shape=self.input_shape,
                                  name='conv1_1'
                              ))
                                     
                                     
        #model.add(ZeroPadding2D((1, 1)))
        #self.cnn_model.add(MaxPooling2D(pool_size=pool_size, strides=(2, 2)))
        #        self.cnn_model.add(Convolution2D(filters=32,
        #                                 kernel_size=(3,3),
        #                                 activation=activation#,
                                         #name='conv1_2'
                                         #                             ))
                                         
        # The pooling layers down-sample the previous layers feature map.
        # - Pooling layers follow a sequence of one or more convolutional layers 
        # and are intended to consolidate the features learned and expressed in the previous layers feature map.
        
        #self.cnn_model.add(MaxPooling2D(pool_size=pool_size))
        
        if batch_norm == True:
            self.cnn_model.add(BatchNormalization(axis=1, momentum=0.99, center=True, scale=True)) # axis=1 when using convolutional layers

        # Dropout: CNNs have a habit of overfitting, even with pooling layers.
        # - Dropout should be used such as between fully connected layers and perhaps after pooling layers.
        self.cnn_model.add(Dropout(dropout_rate))
        self.cnn_model.add(Flatten())
        self.cnn_model.add(Dense(16, activation='sigmoid')) # why 16 instead of say 128?

        #  Dropout: CNNs have a habit of overfitting, even with pooling layers.
        # - Dropout should be used such as between fully connected layers and perhaps after pooling layers.
        self.cnn_model.add(Dropout(dropout_rate))        # saw example where the dropout increased throughout the hidden layers
        if batch_norm == True:
            self.cnn_model.add(BatchNormalization(axis=1, momentum=0.99, center=True, scale=True)) # axis=1 when using convolutional layers

        # Add fully connected output layer with a softmax (or sigmoid) activation function
        self.cnn_model.add(Dense(1, activation='sigmoid'))
        #self.cnn_model.add(Dense(2, activation='softmax'))

        # setting up the optimization of our weights 
        #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

        # Compile neural network (set loss and optimize) 
        self.cnn_model.compile(loss=loss,
                               optimizer=optimizer,
                               metrics=[metric])

        # Print summary report 
        if report_summary:
            self.cnn_model.summary()

        # Store model to file
        self.cnn_model.save(save_as)

        # Return compiled network  
        return self.cnn_model

    # Merged DNN and CNN models
    def build_merged_fn(self,
                        input_dim=12, input_shape=(1, 8, 8),
                        kernel_size=(1,1), # convolution kernel size
                        pool_size=(1, 1),  # size of pooling area for max pooling
                        nlayers=3, list_of_nneurons=[50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
                        dense_layer_sizes=1, nb_filters = 32, # number of convolutional filters to use
                        dropout_rate=0.01, l2_norm=0.001, learning_rate=1e-3, batch_norm=False,
                        activation='relu', kernel_initializer='lecun_normal', optimizer='adam',
                        metric='accuracy', loss='binary_crossentropy'):

        # =========      Mixed deep learning model right branch (DNN)      ======================== 
        branch_right = Sequential()

        # Add fully connected layer with an activation function (input layer) 
        branch_right.add(Dense(units=list_of_nneurons[0],
                               input_dim=self.input_dim,
                               kernel_initializer=kernel_initializer,
                               activation=activation,
                               kernel_regularizer=l2(l2_norm)))
        branch_right.add(Dropout(dropout_rate))

        # Indicate the number of hidden layers
        for index, layer in enumerate(range(self.nlayers-1)):
            self.model.add(Dense(units=list_of_nneurons[index+1],
                                 kernel_initializer=kernel_initializer,
                                 activation=activation,
                                 kernel_regularizer=l2(l2_norm)))
            
        # Add dropout layer
        branch_right.add(Dropout(dropout_rate))

        if batch_norm == True:
            branch_right.add(BatchNormalization())

        # =========      Mixed deep learning model right branch (CNN)      ======================== 
        branch_left = Sequential()

        branch_left.add(Conv2D(filters=nb_filters,            # Number of convolutional filters to use (output feature map)
                               kernel_size=kernel_size,      # Column and Row size of kernel used for convolution
                               padding='valid', #'same'
                               input_shape=self.input_shape,  # 8x8 imagine with 1 channel
                               dim_ordering='tf'
                           ))
        #branch_left.add(Conv2D(filters=nb_filters,
        #                              kernel_size = kernel_size,
        #                              ))
        #branch_left.add(Activation(activation))
        branch_left.add(Conv2D(filters=nb_filters,             # Number of output feature map
                               kernel_size=kernel_size,
                               dim_ordering='tf'
                           ))
        branch_left.add(Activation(activation))
        branch_left.add(MaxPooling2D(pool_size=pool_size))            # Size of pooling area for max pooling
        
        branch_left.add(Flatten()) # Required since merging layer needs matching size
        #branch_left.add(Dense(16, activation='sigmoid'))
        branch_left.add(Dense(128, activation='relu'))
        branch_left.add(Dropout(0.5))
        
        if batch_norm == True:
            branch_left.add(BatchNormalization()) 

        # Merged model
        model = Sequential()

        model.add(Merge([branch_right, branch_left], mode = 'concat'))  # HANDLES the network merging
        model.add(Dense(1, activation="sigmoid", kernel_initializer="normal"))

        #optimizer =  SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)

        model.compile(loss=loss, optimizer='adam', metrics=[metric])

        return model


    # ========================     EXPERIMENTAL SECTION     ======================== 
    def build_inception_fn(self, input_img, lrate = 0.01,  momentum=0.9, epochs = 80):
        '''
        Inception architecture module
        
        Inception modules are mini models inside the bigger model. You do this by doing each convolution in parallel 
        and concatenating the resulting feature maps before going to the next layer. Additionally, this architecture allows 
        the model to recover both local feature via smaller convolutions and high abstracted features with larger convolutions

        Source: https://keras.io/getting-started/functional-api-guide/  
        Source: https://becominghuman.ai/understanding-and-coding-inception-module-in-keras-eb56e9056b4b
        '''

        # Next feed the input tensor to each of the 1x1, 3x3, 5x5 filters in the inception module.
        tower_1 = Conv2D(filters=64, kernel_size=(1,1),   padding='same', activation='relu', kernel_regularizer=l2(0.0002))(input_img)
        tower_1 = Conv2D(filters=64, kernel_size=(3,3),   padding='same', activation='relu')(tower_1)
        
        tower_2 = Conv2D(filters=64, kernel_size=(1,1),   padding='same', activation='relu', kernel_regularizer=l2(0.0002))(input_img)
        tower_2 = Conv2D(filters=64, kernel_size=(5,5),   padding='same', activation='relu')(tower_2)

        #tower_3 = Conv2D(filters=64, (1,1),   padding='same', activation='relu', kernel_regularizer=l2(0.0002))(input_img)
        #tower_3 = Conv2D(filters=64, (7,7),   padding='same', activation='relu')(tower_3)
        tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
        tower_3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.0002))(tower_3)

        #tower_4 = Conv2D(filters=64, (1,1),   padding='same', activation='relu', kernel_regularizer=l2(0.0002))(input_img)
        #tower_4 = Conv2D(filters=64, (8,8),   padding='same', activation='relu')(tower_4)
        tower_4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
        tower_4 = Convolution2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.0002))(tower_4)

        tower_5 = MaxPooling2D((3,3), strides=(1,1),      padding='same')(input_img)
        tower_5 = Conv2D(filters=64, kernel_size=(1,1),   padding='same', activation='relu', kernel_regularizer=l2(0.0002))(tower_5)


        # Note: We learn 64 1x1 filters on the input_img tensor and then we learn 64 3x3 filters on the tower_1 tensor. Similarly, we make tower_2, tower_3 tensors. We provide input_img tensor to tower_2 and tower_3 as input so all the 3x3, 5x5 filters and the max pooling layers are performed on the same input. The padding is kept same so that the output shape of the Conv2D operation is same as the input shape. So, the final output of each filter of tower_1, tower_2 and tower_3 is same. Thus we can easily concatenate these filters to form the output of our inception module.  

        # Concatenate operation assumes that the dimensions of tower_1, tower_2, tower_3, tower_4, tower_5 are the same, except for the concatenation axis.
        output = concatenate([tower_1, tower_2, tower_3, tower_4, tower_5], axis = 3)
        #  We flatten the output to a one dimensional collection of neurons which is then used to create a fully connected neural network as a final classifier
        output = Flatten()(output)
        out    = Dense(2, activation='softmax')(output)

        # Thus we obtain a fully connected neural network with final layer having 1 neurons one corresponding to each class.

        # We can now create the model
        model = Model(inputs = input_img, outputs = out)

        decay = lrate/epochs

        sgd = SGD(lr=lrate, momentum=momentum, decay=decay, nesterov=False)

        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        print(model.summary()) 

        return model


    def build_Alexnet_fn(self, input_shape=(1, 28, 28)):

        '''
        Alexnet architecture
        '''
        # Build model
        model = Sequential()

        #conv1
        model.add(Conv2D(input_shape=input_shape, #(None,None,3) 
                         filters=96, kernel_size=(1,1), #(11, 11),
                         strides=(4,4),
                         padding='valid'
                     )
              )
        model.add(Activation('relu'))
        #pooling1
        model.add(MaxPooling2D(pool_size=(1,1), #(3, 3),
                               strides=(2,2),
                               padding='valid'
                           )
            )

        #conv2
        model.add(Conv2D(filters=256, kernel_size=(5, 5),
                                strides=(1,1),
                                padding='same'
                            )
              )
        model.add(Activation('relu'))
        #pooling2
        model.add(MaxPooling2D(pool_size=(1,1),#(3, 3),
                               strides=(2,2),
                               padding='valid'
                           )
              )

        #conv3
        model.add(Conv2D(filters=384, kernel_size=(3, 3),
                                strides=(1,1),
                                padding='same'
                            )
              )
        model.add(Activation('relu'))

        #conv4
        model.add(Conv2D(filters=384, kernel_size=(3, 3),
                                strides=(1,1),
                                padding='same'
                            )
              )
        model.add(Activation('relu'))

        #conv5
        model.add(Conv2D(filters=256, kernel_size=(3, 3),
                                strides=(1,1),
                                padding='same'
                            )
              )
        model.add(Activation('relu'))
        #pooling3
        model.add(MaxPooling2D(pool_size=(1,1),#(3, 3),
                               strides=(2,2),
                               padding='valid'
                           )
              )

        #conv6
        model.add(Conv2D(filters=4096, kernel_size=(1,1),#(6, 6),
                                strides=(1,1),
                                padding='valid'
                            )
              )
        model.add(Activation('relu'))

        #conv7
        model.add(Conv2D(filters=4096, kernel_size=(1, 1),
                                strides=(1,1),
                                padding='valid'
                            )
              )
        model.add(Activation('relu'))
        
        #conv8
        model.add(Conv2D(filters=2, kernel_size=(1, 1),
                                strides=(1,1),
                                padding='valid'
                            )
              )
        model.add(Flatten())
        #model.add(GlobalAveragePooling2D())
        model.add(Dense(1, activation='sigmoid'))
        #model.add(Activation('softmax'))
                
        model.compile(loss='binary_crossentropy', #'categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model

    # Create the model
    def build_BrownLeeNet_fn(self, input_shape, dropout_rate=0.2, momentum=0.9, nesterov=True, epochs = 80, lrate = 0.01):


        model = Sequential()

        model.add(Conv2D(input_shape=input_shape, 
                         filters=32, kernel_size=(2, 2), 
                         activation='relu', padding='same'))
        #model.add(Dropout(dropout_rate))
        model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu', padding='same')) 
        model.add(MaxPooling2D(pool_size=(1, 1)))
        model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu', padding='same'))
        #model.add(Dropout(dropout_rate))
        model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu', padding='same')) 
        model.add(MaxPooling2D(pool_size=(1, 1)))
        model.add(Conv2D(filters=128, kernel_size=(2, 2), activation='relu', padding='same'))
        #model.add(Dropout(dropout_rate))
        model.add(Conv2D(filters=128, kernel_size=(2, 2), activation='relu', padding='same')) 
        model.add(MaxPooling2D(pool_size=(1, 1)))
        model.add(Flatten())
        #model.add(Dropout(dropout_rate))
        model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(1))) #maxnorm(3)
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(1))) #maxnorm(3)
        #model.add(Dropout(dropout_rate))
        
        # Ouput layer
        model.add(Dense(1, activation='sigmoid'))

        decay = lrate/epochs
        sgd = SGD(lr=lrate, momentum=momentum, decay=decay, nesterov=nesterov)

        # Compile model
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy']) 
        
        print(model.summary())
        
        return model


    def build_AtlasNet_fn(self, x):
        '''
        source: https://github.com/eracah/atlas_dl/blob/micky/train.py
        source: https://www.nersc.gov/users/data-analytics/data-analytics-2/deep-learning/deep-networks-for-hep
        '''

        h = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
        h = BatchNormalization()(h)             # comment out (AUC: 0.647852)
        h = MaxPooling2D(pool_size=(1, 1))(h)   # comment out (AUC: 0.647852)
        h = Dropout(0.5)(h)                     # comment out (AUC: 0.647852)
        h = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', strides=2, padding='same')(h)
        h = BatchNormalization()(h)             # comment out (AUC: 0.657021)
        h = MaxPooling2D(pool_size=(1, 1))(h)   # comment out  (AUC: 0.666366)  init: (2, 2) 
        h = Dropout(0.5)(h)                     # comment out (AUC: 0.657021)  
        h = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(h)
        #h = BatchNormalization()(h)            # comment out (AUC: 0.652616)
        #h = MaxPooling2D(pool_size=(1, 1))(h)  # comment out (AUC: 0.657565)   init: (2, 2)   
        h = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', strides=2, padding='same')(h)
        #h = BatchNormalization()(h)            # comment out (AUC: 0.656422)

        h = Flatten()(h)
        h = Dense(512, activation='relu')(h)
        #h = Dropout(0.5)(h)                     # comment out (AUC: 0.644624) # remove
        y = Dense(2, activation='sigmoid')(h)

        model = Model(inputs=x, outputs=y)
        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model


    def build_conv_autoencoder_fn(self, input_dim):
        '''
        Convolutional autoencoder: generative model
        source: https://blog.keras.io/building-autoencoders-in-keras.html
                Info: https://jhui.github.io/2018/02/11/Keras-tutorial/
        '''
        # adapt this if using `channels_first` image data format

        input_img = Input(shape=input_dim)
        # =========      ENCODER     ========================  
        x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_img)  #filters=16
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)          #filters=8
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)          #filters=8
        encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        print("shape of encoded", K.int_shape(encoded))

        # =========      DECODER     =========================
        x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(encoded)    #filters=8
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)          #filters=8
        x = UpSampling2D((2, 2))(x)
        
        # In original tutorial, padding='same' was used.
        # then the shape of 'decoded' will be 32 x 32, instead of 28 x 28
        x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='valid')(x)         #filters=32
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(filters=1, kernel_size=(5, 5), activation='sigmoid', padding='same')(x)
        print("shape of decoded", K.int_shape(decoded))
        
        # - Build the encoder pipeline 
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        return autoencoder


    def build_Unet_fn(self, nClasses, inputs, optimizer=None , input_width=360 , input_height=480 , nChannels=1 ):
        
        '''
        source: https://www.programcreek.com/python/example/89691/keras.layers.UpSampling2D
        Example 15
        '''

        #inputs = Input((nChannels, input_height, input_width))
        conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
        conv1 = Dropout(0.2)(conv1)
        conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
        conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
        conv2 = Dropout(0.2)(conv2)
        conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
        conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
        conv3 = Dropout(0.2)(conv3)
        conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
        
        up1 = merge([UpSampling2D(size=(2, 2))(conv3), conv2], mode='concat', concat_axis=1)
        conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up1)
        conv4 = Dropout(0.2)(conv4)
        conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
    
        up2 = merge([UpSampling2D(size=(2, 2))(conv4), conv1], mode='concat', concat_axis=1)
        conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up2)
        conv5 = Dropout(0.2)(conv5)
        conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv5)
    
        conv6 = Convolution2D(nClasses, 1, 1, activation='relu',border_mode='same')(conv5)
        conv6 = core.Reshape((nClasses,input_height*input_width))(conv6)
        conv6 = core.Permute((2,1))(conv6)

        conv7 = core.Activation('softmax')(conv6)

        model = Model(input=inputs, output=conv7)

        if not optimizer is None:
            model.compile(loss="categorical_crossentropy", optimizer= optimizer , metrics=['accuracy'] )
            
        return model 

    def build_fcn_fn(self, x):
        '''
        FCN full connected network
        https://github.com/eracah/atlas_dl/blob/micky/train.py
        '''

        h = Dense(2048, activation='relu')(x)
        h = Dropout(0.5)(h)
        h = Dense(2048, activation='relu')(h)
        h = Dropout(0.5)(h)
        h = Dense(2048, activation='relu')(h)
        h = Dropout(0.5)(h)
        h = Dense(2048, activation='relu')(h)
        h = Dropout(0.5)(h)
        h = Dense(2048, activation='relu')(h)
        h = Dropout(0.5)(h)
        h = Dense(2048, activation='relu')(h)
        h = Dropout(0.5)(h)
        h = Dense(2048, activation='relu')(h)
        h = Dropout(0.5)(h)
        h = Dense(2048, activation='relu')(h)
        h = Dropout(0.5)(h)
        y = Dense(1, activation='sigmoid')(h)

        model = Model(inputs = x, outputs = y)

        model.summary()

        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

        return model
    # ========================================================================================


    # Get model
    def estimator(self):
        '''
        Estimator
        '''
        return self.model

    # Define get parameters
    def get_params(self):
        '''
        Get parameters
        '''
        return {'input_dim': self.input_dim, 'nlayers': self.nlayers, 'nneurons': self.nneurons,
                'droupout': self.dropout_rate, 'l2_norm': self.l2_norm, 'learning_rate': self.learning_rate,
                'activation': self.activation, 'kernel_initializer': self.kernel_initializer, 
                'optimizer': self.optimizer, 'metric': self.metric, 'loss': self.loss, 
                'report_summary': self.report_summary}



    ## Store architecture as JSON file and weights separately      
    def save_model(self, model, filename='model.h5'):
        '''
        save_model" store keras model (architecture + weights + optimizer state) in one output file    
        '''
        print("Saved model to disk")
        # creates a HDF5 file 'model.h5'
        return model.save(filename)

    def serialize_model_architecture(self, model, filename='model.json'):
        '''
        serialize_model_architecture: serialize model's architecture to JSON
        ''' 
        model_json = model.to_json()
        with open(filename, "w") as json_file:
            json_file.write(model_json)

        print("Saved model to disk")
        return json_file

    def serialize_model_weights(self, model, filename='model_weights.h5', overwrite=False):
        '''
        serialize_model_weights: serialize only model's weights to HDF5    
        '''
        model.save_weights(filename, overwrite=overwrite)
        print("Saved model to disk")
        return model

    # Helper function to simplify the construction of Conv2D followed by a activation function (e.g. ReLu) 
    # followed by a BatchNormalization
    def Conv2DReluBatchNorm(self, n_filter, w_filter, h_filter, inputs):
        activation='relu'
        return BatchNormalization()(Activation(activation=activation)(Convolution2D(n_filter, 
                                                                                    w_filter, 
                                                                                    h_filter, 
                                                                                    padding='same')(inputs)))

    # Helper function to visualize the shape of the weights learned by each layer.
    def print_network_weights(model):
        '''
        Print network works per layer
        '''
        for i, layer in enumerate(model.layers):
            if len(layer.get_weights()) > 0:

                W, b = layer.get_weights()

                print("Layer", i, "\t", layer.name, "\t\t", W.shape, "\t", b.shape)



class ResnetBuilder(object):

    def __init__(self):
        pass

    @staticmethod
    def _handle_dim_ordering():
        global ROW_AXIS
        global COL_AXIS
        global CHANNEL_AXIS

        if K.image_dim_ordering() == 'tf':
            ROW_AXIS = 1
            COL_AXIS = 2
            CHANNEL_AXIS = 3
        else:
            CHANNEL_AXIS = 1
            ROW_AXIS = 2
            COL_AXIS = 3

    @staticmethod
    def _bn_relu(input):
        """
        Helper to build a BatchNormalization -> relu block
        """
        norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
        return Activation("relu")(norm)

    @staticmethod
    def _conv_bn_relu(**conv_params):
        """
        Helper to build a conv -> BatchNormalization -> relu block
        """
        filters            = conv_params["filters"]
        kernel_size        = conv_params["kernel_size"]
        strides            = conv_params.setdefault("strides", (1, 1))
        kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
        padding            = conv_params.setdefault("padding", "same")
        kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

        def f(input):
            conv = Conv2D(filters=filters, kernel_size=kernel_size,
                          strides=strides, padding=padding,
                          kernel_initializer=kernel_initializer,
                          kernel_regularizer=kernel_regularizer)(input)
            return ResnetBuilder._bn_relu(conv)

        return f

    @staticmethod
    def _bn_relu_conv(**conv_params):
            """
            Helper to build a BatchNormalization -> relu -> conv block.
            This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
            """
            filters            = conv_params["filters"]
            kernel_size        = conv_params["kernel_size"]
            strides            = conv_params.setdefault("strides", (1, 1))
            kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
            padding            = conv_params.setdefault("padding", "same")
            kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

            def f(input):
                activation = ResnetBuilder._bn_relu(input)
                return Conv2D(filters=filters, kernel_size=kernel_size,
                              strides=strides, padding=padding,
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=kernel_regularizer)(activation)

            return f

    @staticmethod
    def _shortcut(input, residual):
        """
        Adds a shortcut between input and residual block and merges them with 'sum'
        """
        # Expand channels of shortcut to match residual.
        # Stride appropriately to match residual (width, height)
        # Should be int if network architecture is correctly configured.
        input_shape    = K.int_shape(input)
        residual_shape = K.int_shape(residual)
        stride_width   = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
        stride_height  = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
        equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

        shortcut = input
        # 1 X 1 conv if shape is different. Else identity.
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                              kernel_size=(1, 1),
                              strides=(stride_width, stride_height),
                              padding="valid",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(0.0001))(input)

        return add([shortcut, residual])

    @staticmethod
    def _residual_block(block_function, filters, repetitions, is_first_layer=False):
        """
        Builds a residual block with repeating bottleneck blocks.
        """
        def f(input):
            for i in range(repetitions):
                init_strides = (1, 1)
                if i == 0 and not is_first_layer:
                    init_strides = (2, 2)
                    input = block_function(filters=filters, init_strides=init_strides,
                                           is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
            return input
        return f
        
    @staticmethod
    def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
        """
        Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
        Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
        """
        def f(input):
            if is_first_block_of_first_layer:
                # don't repeat bn->relu since we just did bn->relu->maxpool
                conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                               strides=init_strides,
                               padding="same",
                               kernel_initializer="he_normal",
                               kernel_regularizer=l2(1e-4))(input)
            else:
                conv1 = ResnetBuilder._bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                      strides=init_strides)(input)

                residual = ResnetBuilder._bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
            return ResnetBuilder._shortcut(input, residual)
        return f

    @staticmethod
    def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
        """
        Bottleneck architecture for > 34 layer resnet.
        Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
        Returns:
        A final conv layer of filters * 4
        """
        def f(input):
            if is_first_block_of_first_layer:
                # don't repeat bn->relu since we just did bn->relu->maxpool
                conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                                  strides=init_strides,
                                  padding="same",
                                  kernel_initializer="he_normal",
                                  kernel_regularizer=l2(1e-4))(input)
            else:
                conv_1_1 = ResnetBuilder._bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                                       strides=init_strides)(input)

                conv_3_3 = ResnetBuilder._bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
                residual = ResnetBuilder._bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
            return ResnetBuilder._shortcut(input, residual)
        return f

    @staticmethod
    def _get_block(identifier):
        if isinstance(identifier, six.string_types):
            res = globals().get(identifier)
            if not res:
                raise ValueError('Invalid {}'.format(identifier))
            return res
        return identifier

    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        """
        Builds a custom ResNet like architecture.
        Args:
        input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
        num_outputs: The number of outputs at final softmax layer
        block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
            The original paper used basic_block for layers < 50
        repetitions: Number of repetitions of various block units.
            At each block unit, the number of filters are doubled and the input size is halved
        Returns:
        The keras `Model`.
        """
        ResnetBuilder._handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = ResnetBuilder._get_block(block_fn)

        input = Input(shape=input_shape)
        conv1 = ResnetBuilder._conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)
            
        block = pool1
        filters = 64

        for i, r in enumerate(repetitions):
            block = ResnetBuilder._residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = ResnetBuilder._bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(flatten1)

        model = Model(inputs=input, outputs=dense)

        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, ResnetBuilder.basic_block, [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, ResnetBuilder.basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, ResnetBuilder.bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, ResnetBuilder.bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, ResnetBuilder.bottleneck, [3, 8, 36, 3])

