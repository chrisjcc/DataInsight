# ---- Basic import libraries
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# ---- Import numpy and pandas libraries
import numpy
import numpy as np
import pandas as pd

# ---- Import from root_numpy library 
#import root_numpy
#from root_numpy import root2array, rec2array, tree2array

# ---- Import from root_pandas library
#import root_pandas
#from root_pandas import read_root

# ---- Import from keras
from keras.utils import np_utils
from keras import backend as K

# ---- Import UpRoot
import uproot

# ---- Import ROOT library
#from ROOT import TVector3, TLorentzVector, TChain, TFile

# ---- Import additional library
import re
import collections

from scipy.misc import bytescale

# ----- Loader base class
class loader(object):
    def __init__(self):
        pass
    def __call__(self):
        pass

# ---- Data loader class
class DataLoader(loader):

    # Class constructor initialize fields
    def __init__(self, sig_filename, bkg_filename, treename, features, weight=None):

        self.sig_filename=sig_filename
        self.bkg_filename=bkg_filename
        self.treename=treename
        self.features=features
        self.weight=weight

        self.data = self.__call__(sig_filename=self.sig_filename, 
                                  bkg_filename=self.bkg_filename, 
                                  treename=self.treename, 
                                  features=self.features, 
                                  weight=self.weight)

    # Function to convert ROOT files into pandas dataframe
    def __call__(self, sig_filename, bkg_filename, treename, features, weight):
        """Data loader.

        Parameters
        ----------
        sig_filename : array, shape = [n_samples]
        true class, intergers in [0, n_classes - 1)
        bkg_filename : array, shape = [n_samples, n_classes]
        category: string
        features: array, shape = [n_features]

        Returns
        -------
        data : pandas.DataFrame
        """

        # Reader tree for signal
        signal_tree     = uproot.open(sig_filename)[treename]
        background_tree = uproot.open(bkg_filename)[treename]

        sig = signal_tree.pandas.df(features)
        bkg = background_tree.pandas.df(features)

        # Set class label
        sig['y'] = 1.
        bkg['y'] = 0.

        # Merge signal and & background dataframes
        data = pd.concat([sig, bkg])

        return data

    # Fucntion to create two-dimensional image from pandas dataframe used in CNN modeling
    def create_image(self, data, obj, input_shape, image_data_format='channels_first'):
        '''
        create_image: converts pandas dataframe into two-dimensional image
        Source: code from James Keaveney (current version can only handle flat trees, possibly workaround using the expand_arrays class method) 
        '''
        print('input_shape: ', input_shape)

        # Set eta-phi bin ranges
        # trying 16 bins in eta, phi to see if there is any improvment.
        #eta_bins = np.array([-2.5,-2.1875,-1.875,-1.5625,-1.25,-0.9375,-0.625,-0.3125,0.0,0.3125,0.625,0.9375,1.25,1.5625,1.875,2.1875, 2.5])
        #phi_bins = np.array([-3.2, -2.8000000000000003, -2.4000000000000004, -2.0, -1.6, -1.2000000000000002, -0.7999999999999998, -0.3999999999999999, 0.0, 0.3999999999999999, 0.7999999999999998, 1.2000000000000002, 1.6000000000000005, 2.0, 2.4000000000000004, 2.8, 3.2])
        eta_bins = np.array([-2.5, -1.875, -1.25, -0.625, 0.0, 0.625, 1.25, 1.875, 2.5])
        phi_bins = np.array([-3.2, -2.4, -1.6, -0.8, 0.0, 0.8, 1.6, 2.4, 3.2])
                
        obj_map_bins = np.array([eta_bins, phi_bins], dtype=np.float64)
        obj_map      = np.zeros((input_shape[1],  input_shape[2]), dtype=np.float64)
        obj_maps     = np.zeros((len(data.index), input_shape[1], input_shape[2]), dtype=np.float64)
        
        obj_etas = np.array([6], dtype=np.float64)
        obj_phis = np.array([6], dtype=np.float64)
        obj_pts  = np.array([6], dtype=np.float64)

        objectCollection = [] 

        for i, (index, row) in enumerate(data.iterrows()):
            obj_map  = np.zeros((input_shape[1], input_shape[2]), dtype=np.float64)
            obj_etas = np.zeros((6), dtype=np.float64)
            obj_phis = np.zeros((6), dtype=np.float64)
            obj_pts  = np.zeros((6), dtype=np.float64)

            if str(obj) is 'selJet':
                for obj_index in xrange(len(row[str(obj)])):
                    
                    tlv = TLorentzVector(row[str(obj)][obj_index][0], row[str(obj)][obj_index][1],
                                         row[str(obj)][obj_index][2], row[str(obj)][obj_index][1])
                
                    objectCollection.append(tlv)


            elif str(obj) is 'sel_lep':
                for obj_index in xrange(len(row[str(obj)])):

                    tlv = TLorentzVector(row[str(obj)][obj_index][0], row[str(obj)][obj_index][1],
                                         row[str(obj)][obj_index][2], row[str(obj)][obj_index][1])

                    objectCollection.append(tlv)

            elif str(obj) is 'jet':
                v1 = TVector3(row[obj+'_1_px'], row[obj+'_1_py'], row[obj+'_1_pz']) 
                v2 = TVector3(row[obj+'_2_px'], row[obj+'_2_py'], row[obj+'_2_pz'])
                v3 = TVector3(row[obj+'_3_px'], row[obj+'_3_py'], row[obj+'_3_pz'])
                v4 = TVector3(row[obj+'_4_px'], row[obj+'_4_py'], row[obj+'_4_pz'])
                v5 = TVector3(row[obj+'_5_px'], row[obj+'_5_py'], row[obj+'_5_pz'])
                v6 = TVector3(row[obj+'_6_px'], row[obj+'_6_py'], row[obj+'_6_pz'])

                obj_phis = [v1.Phi(), v2.Phi(), v3.Phi(), v4.Phi(), v5.Phi(), v6.Phi()]
                obj_etas = [v1.Eta(), v2.Eta(), v3.Eta(), v4.Eta(), v5.Eta(), v6.Eta()]
                obj_pts  = [v1.Pt(),  v2.Pt(),  v3.Pt(),  v4.Pt(),  v5.Pt(),  v6.Pt()]
                
            elif str(obj) is 'bjet':
                v1 = TVector3(row[obj+'_1_px'], row[obj+'_1_py'], row[obj+'_1_pz'])
                v2 = TVector3(row[obj+'_2_px'], row[obj+'_2_py'], row[obj+'_2_pz'])
                v3 = TVector3(row[obj+'_3_px'], row[obj+'_3_py'], row[obj+'_3_pz'])
                v4 = TVector3(row[obj+'_4_px'], row[obj+'_4_py'], row[obj+'_4_pz'])
                v5 = TVector3(row[obj+'_5_px'], row[obj+'_5_py'], row[obj+'_5_pz'])
                v6 = TVector3(row[obj+'_6_px'], row[obj+'_6_py'], row[obj+'_6_pz'])
                
                obj_phis = [v1.Phi(), v2.Phi(), v3.Phi(), v4.Phi(), v5.Phi(), v6.Phi()]
                obj_etas = [v1.Eta(), v2.Eta(), v3.Eta(), v4.Eta(), v5.Eta(), v6.Eta()]
                obj_pts  = [v1.Pt(),  v2.Pt(),  v3.Pt(),  v4.Pt(),  v5.Pt(),  v6.Pt()]

            elif str(obj) is "lepton":
                v1 = TVector3(row[obj+'_1_px'], row[obj+'_1_py'], row[obj+'_1_pz'])
                v2 = TVector3(row[obj+'_2_px'], row[obj+'_2_py'], row[obj+'_2_pz'])

                obj_phis = [v1.Phi(), v2.Phi()]
                obj_etas = [v1.Eta(), v2.Eta()]
                obj_pts  = [v1.Pt(),  v2.Pt()]

            elif isinstance(row[obj+'_pt'], numpy.ndarray):
                tv3 = TVector3()
                nobjs_per_sample = len(row[obj+'_pt'])
                
                for obj_index in xrange(nobjs_per_sample):
                    tv3.SetPtEtaPhi(row[obj+'_pt'][obj_index], row[obj+'_eta'][obj_index], row[obj+'_phi'][obj_index])
                    objectCollection.append(tv3)

                obj_phis = [v.Phi() for v in objectCollection]
                obj_etas = [v.Eta() for v in objectCollection]
                obj_pts  = [v.Pt()  for v in objectCollection]

            inds_phi = np.digitize(obj_phis, phi_bins)
            inds_eta = np.digitize(obj_etas, eta_bins)
            
            # Fill obj-map
            for obj_i in range(0, len(obj_pts)):
                obj_map[inds_eta[obj_i] -1, inds_phi[obj_i] -1 ] = obj_pts[obj_i]

                if ((i == 0) & (obj == "lepton" or obj == "jet")):
                    print("adding "+str(obj)+" to map with eta, phi, pt "+ str(obj_etas[obj_i])+ " , " 
                           + str(obj_phis[obj_i]) + " "+str(obj_pts[obj_i]) + " to map number " + str(i))

            # Set object map
            obj_maps[i] = obj_map

        # Transform dataset from having shape (n, width, height) to (n, depth, width, height)
        if  image_data_format == 'channels_first':
            obj_maps = obj_maps.reshape(obj_maps.shape[0], 1, input_shape[1], input_shape[2])
        elif image_data_format == 'channels_last':
            obj_maps = obj_maps.reshape(obj_maps.shape[0], input_shape[1], input_shape[2], 1)

        print("Keras backend image data format is: ", K.image_data_format())
        print("Object maps shape =  ", str(obj_maps.shape))

        return obj_maps


    # Function to handle pandas data frame feature column with array to expand to spearate column
    def expand_arrays(self, data):
        '''
        expand_arrays: method flattens pandas dataframe data columns with array structure

        Source: https://github.com/aelwood/pandasPlotting/blob/45af2d118a798201c708238b3dc01abf6a1b9170/dfFunctions.py 
        Seepd up!: https://stackoverflow.com/questions/38203352/expand-pandas-dataframe-column-into-multiple-rows
        '''
        for feature in data.keys():

            # Check if it's a list
            if data[feature].dtype.kind is 'O': # g.e kind: i (int), f (float),  O (object)
                    
                # Assuming it's a row, so expand it out
                data_flatten = data[feature].apply(pd.Series)
                data_flatten.columns = [feature+str(key) for key in data_flatten.keys()]
                    
                # Drop the old feature columns
                data = data.drop([feature], axis=1)
                
                # Concatenate original dataset with the flatten columns
                data = pd.concat([data, data_flatten], axis=1)


                # Replace all NaN elements with 0s
                data = data.fillna(0.)

        return data
        
    # Function to get sample (i.e. event) weights
    def sample_weights(data, weight_name = 'globalTimesEventWeight'):

        # calculate event weights, normalized to 1 for signal and background
        #weight_sig = 100000./len(signal.index)
        #weight_bkg = 100000./len(background.index)
        #print("Sample weight: ", weight_sig/weight_bkg)
        #signal['weight'] = np.full(len(signal.index), weight_sig)
        #background['weight'] = np.full(len(background.index), weight_bkg)

        weight_sig = data.query("y == 1.")[weight_name]
        weight_bkg = data.query("y == 0.")[weight_name]

