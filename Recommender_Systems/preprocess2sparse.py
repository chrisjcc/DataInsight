from __future__ import print_function, division
from builtins import range, input

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse import save_npz, load_npz
from sklearn.utils import shuffle


# load data
data = pd.read_csv('../data/edited_rating.csv')

# set number of user and movies
N = data.userId.max() + 1 # number of users
M = data.movie_idx.max() + 1 # number of movies


# split data into train and test
data   = shuffle(data)
cutoff = int(0.8*len(data))

data_train = data.iloc[:cutoff]
data_test  = data.iloc[cutoff:]

R = lil_matrix((N, M))
print("calling: update_train")


count = 0
def update_train(row):
  global count
  count += 1

  if count % 100000 == 0:
    print("processed: %.3f" % (float(count)/cutoff))

  i = int(row.userId)
  j = int(row.movie_idx)
  R[i,j] = row.rating

data_train.apply(update_train, axis=1)

# mask, to tell which entries exist and which do not
R = R.tocsr()
mask = (R > 0)
save_npz("rating_train.npz", R)


# test ratings dictionary
R_test = lil_matrix((N, M))
print("calling: update_test")

count = 0
def update_test(row):
  global count
  count += 1

  if count % 100000 == 0:
    print("processed: %.3f" % (float(count)/len(data_test)))

  i = int(row.userId)
  j = int(row.movie_idx)
  R_test[i,j] = row.rating

data_test.apply(update_test, axis=1)

R_test = R_test.tocsr()
mask_test = (R_test > 0)
save_npz("rating_test.npz", R_test)
