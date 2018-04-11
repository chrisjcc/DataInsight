# ----- Import basic libraries
from __future__ import print_function
import numpy as np
import random

import pandas

# Import scikit-learn
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
from sklearn.model_selection import cross_val_score

# Hyper-paramter tuning using HyperOpt library
from hyperopt.pyll import scope
from hyperopt import hp
from hyperopt import fmin, Trials, STATUS_OK, tpe, space_eval

# ---- Scikit-Learn Optimizer
from skopt import gp_minimize, forest_minimize
from skopt.plots import plot_convergence
from skopt.plots import plot_evaluations

# ---- Optimize bayesian
from bayes_opt import BayesianOptimization

# ---- Python utilities
from collections import defaultdict, Counter

# Fix random seed for reproducibility
seed = 7
random.seed(a=seed)


## Standard Nested K-fold Cross Validation
class CrossValidation(object):
    '''
    Nested K-fold Cross Validation
    '''
    # Class constructor
    def __init__(self):
        pass
    
    # Define Nested K-fold Cross Validation function
    def nested_grid_search_cv(self, model, X, y, outer_cv, inner_cv,
                              param_grid, scoring="accuracy",
                              n_jobs=1):
        """
        Nested k-fold crossvalidation.
        
        Parameters
        ----------
        Classifier : array, shape = [n_samples]
        true class, intergers in [0, n_classes - 1)
        X : array,  shape = [n_samples, n_classes]
        y : array,  shape = [n_samples, n_classes]
        outer_cv:   shape = [n_samples, n_classes]
        inner_cv:   shape = [n_samples, n_classes]
        param_grid: shape = [n_samples, n_classes]
        scoring:    shape = [n_samples, n_classes]
        
        Returns
        -------
        Grid classifier: classifier re-fitted to full dataset
        
        grid_search: GridSearchCV object
        A post-fit (re-fitted to full dataset) GridSearchCV object where the estimator is a Pipeline.
        """

        outer_scores = []

        # Set up grid search configuration
        grid =  GridSearchCV(estimator=model, param_grid=param_grid,
                             cv=inner_cv, scoring=scoring, n_jobs=n_jobs)
        
        # Set aside a hold-out test dataset for model evaluation
        for k, (training_samples, test_samples) in enumerate(outer_cv.split(X, y)):

            # x training and test datasets
            if isinstance(X, pandas.core.frame.DataFrame):
                x_train = X.iloc[training_samples]
                x_test  = X.iloc[test_samples]
            else:  # in case of spare matrices
                x_train = X[training_samples]
                x_test  = X[test_samples]

            # y training and test datasets
            if isinstance(y, pandas.core.frame.Series):
                y_train = y.iloc[training_samples]
                y_test  = y.iloc[test_samples]
            else: # in case of numpy arrays
                y_train = y[training_samples]
                y_test  = y[test_samples]

            # Build classifier on best parameters using outer training set
            # Fit model to entire training dataset (i.e tuning & validation dataset)
            print("fold-%s model fitting ..." % (k+1))

            # Train on the training set
            grid.fit(x_train, y_train)

            # Check model stability
            print(grid.best_estimator_.get_params())

            # Evaluate
            score = grid.score(x_test, y_test)

            outer_scores.append(score)
            print("\tModel validation score", score)

        # Print final model evaluation (i.e. mean cross-validation scores)
        print("Final model evaluation (mean cross-val scores):\n", np.array(outer_scores).mean())
            
        # Note: the scoring is being done without the weights associated with X
        # Fit model to entire training dataset (i.e tuning & validation dataset)
        print("Performing fit over entire training data\n")
        grid.fit(X, y)

        return grid


## Handles objective function for HyperOptimization
class HyperOptObjective(object):
    '''
    HyperOptimization
    '''
    # Class constructor initialize data members
    def __init__(self, X, y, n_folds=3, scoring='roc_auc', n_jobs=1):
        self.X = X
        self.y = y
        self.seed = 42
        self.n_folds= n_folds
        self.cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.best = 0

    # Define the objective function
    def __call__(self, params):
        auc_score = self.hyperopt_train_test(params)

        # Involving accuracy
        #if auc_score > self.best:
        #    self.best = acc
        #print('new best:', self.best, params)
        #return {'loss': -auc_score, 'status': STATUS_OK}
        print('SCORE:', auc_score, params)
        return {'loss': 1-auc_score, 'status': STATUS_OK }

    # Handles the evaluation to be minimized
    def hyperopt_train_test(self, params):
        self.classifier.set_params(**params)

        score = cross_val_score(estimator=self.classifier, X=self.X, y=self.y,
                                cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs).mean()

        # Note: hyperopt works by minimizing an objective function and sampling parameter values from various distributions.
        # Since the goal is to maximize the AUC score, we will have hyperopt minimize 1 - AUC.
        return 1 - score # or -score


    # Set the classifier algorithm
    def setEstimator(self, classifier):
        self.classifier = classifier
        return self.classifier


## Handles objective function for BayesianOptimization
class BayesOptObjective(object):
    '''
    BayesianOptimization
    '''
    # Class constructor initialize data members
    def __init__(self, Classifier, X, y, n_folds=3, scoring='roc_auc', n_jobs=1):
        self.X = X
        self.y = y
        self.seed = 42
        self.n_folds= n_folds
        self.cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        self.scoring = scoring
        self.n_jobs=n_jobs
        self.Classifier = Classifier
        self.hyper_parameter = {}

    # Define the objective function
    def __call__(self, **params):

        for key, value in self.hyper_parameter.iteritems():

            if self.get_type(key) == 'int':
                params[key] = int(params[key])
            elif self.get_type(key) == 'float':
                params[key] = float(params[key])

        self.Classifier.set_params(**params)

        score = cross_val_score(estimator=self.Classifier, X=self.X, y=self.y,
                                scoring=self.scoring, cv=self.cv, n_jobs=1).mean()

        return 1-score

    def set_type(self, parameter_key, parameter_type):
        self.hyper_parameter[parameter_key] = parameter_type
        return self.hyper_parameter

    def get_type(self, parameter_key):
        return self.hyper_parameter[parameter_key]


## Handles objective function for Scikit-learn optimization:
# Bayesian optimization based on Gaussian process regression search (controlling the exploration-exploitation trade-off)
class SkOptObjective(object):
    '''
    Scikit-learn optimization
    '''
    # class constuctor
    def __init__(self, estimator, X, y, n_folds=3, scoring='roc_auc', n_jobs=1, seed=42):
        # Split development set into a train and test set
        self.X = X
        self.y = y
        self.seed = seed
        self.n_folds= n_folds
        self.cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.clf = estimator
 
    # objective function we want to minimize 
    def __call__(self, values):
        # Build paramter list with updated values
        params_dict = dict(zip(self.keys, values))

        # Get sample weights
        #weights = self.X['weight'].values

        # Optimize hyper-parameters of classifier
        self.clf.set_params(**params_dict)

        # Cross-validation mean absolute error of a classifier as a function of its hyper-parameters
        score = cross_val_score(self.clf, 
                                #self.X.drop(['weight'], axis=1, inplace=False),
                                self.X,
                                self.y, 
                                cv=self.cv, 
                                scoring=self.scoring, 
                                n_jobs=self.n_jobs, 
                                #fit_params={'kerasclassifier__sample_weight': weights}
                                ).mean()
        return 1-score #-score
    
    # Set classifier
    def setEstimator(self, estimator):
        self.clf = estimator

    # Set list features 
    def paramKeys(self, keys):
        self.keys = keys

