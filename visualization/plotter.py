## Plotter module
from __future__ import print_function

# Import matplotlib library
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import seaborn as sns
import numpy as np
import pandas as pd

# Import panda library
from pandas.tools import plotting
from pandas.tools.plotting import scatter_matrix
from pandas.core.index import Index
import pandas.core.common as com

# Import scipy
import scipy
from scipy.stats import ks_2samp
import scipy as sp
from scipy.stats import distributions
from scipy import interp

# Import scikit-learn
import sklearn
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                             auc, average_precision_score, precision_score,
                             brier_score_loss, recall_score, f1_score, log_loss,
                             classification_report, precision_recall_curve)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

# Import itertools
import itertools
from itertools import cycle

# Import basic library
import os
import joblib
from copy import deepcopy

## Plotter class
class Plotter(object):

    # Class constructor
    def __init__(self):
        pass

    # Correlation matrix of input variables
    def correlation_matrix(self, data, **kwds):
        """Multi class version of Logarithmic Loss metric.
        https://www.kaggle.com/wiki/MultiClassLogLoss

        Parameters
        ----------
        data : array, shape = [n_samples]
        true class, intergers in [0, n_classes - 1)
        kwds : array, shape = [n_samples, n_classes]
        
        Returns
        -------
        loss : float
        """
        
    
        """To calculate pairwise correlation between features.
    
        Extra arguments are passed on to DataFrame.corr()
        """
    
        # Select signal or background label for plot title
        if (data["y"] > 0.5).all(axis=0):
            self.label = "signal"
        elif (data["y"] < 0.5).all(axis=0):
            self.label = "background"
    
        # simply call df.corr() to get a table of
        # correlation values if you do not need
        # the fancy plotting
        self.data = data.drop("y", axis=1) 
 
        # Add colorbar, make sure to specify tick locations to match desired ticklabels
        self.labels = self.data.corr(**kwds).columns.values
    
        self.fig, self.ax1 = plt.subplots(ncols=1, figsize=(9,8))
    
        opts = {"annot" : True,
                "ax" : self.ax1,
                "vmin": 0, "vmax": 1*100,
                "annot_kws" : {"size": 8}, 
                "cmap": plt.get_cmap("Blues", 20),
                }
    
        self.ax1.set_title("Correlations: " + self.label)

        sns.heatmap(self.data.corr(method="spearman").iloc[::-1]*100, **opts) 
        
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)
    
        for ax in (self.ax1,):
            # shift location of ticks to center of the bins
            ax.set_xticks(np.arange(len(self.labels))+0.5, minor=False)
            ax.set_yticks(np.arange(len(self.labels))+0.5, minor=False)
            ax.set_xticklabels(self.labels[::-1], minor=False, ha="right", rotation=70)
            ax.set_yticklabels(np.flipud(self.labels), minor=False)
        
        plt.tight_layout()
        
        return plt.show()
      
    ## Confusion matrix
    def plot_confusion_matrix(self, y_true, y_pre, class_names, 
                              normalize=True, title='Normalized confusion matrix', 
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_true, y_pred)

        # Plot non-normalized confusion matrix (or normalized)
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=normalize, title=title)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        self.fig, self.ax1 = plt.subplots(ncols=1, figsize=(10, 10)) #test

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        return plt.show()
      

    ## Defined overfitting plot
    def overfitting(self, estimator, X_train, X_test, y_train, y_test, bins=50):
        """
        Multi class version of Logarithmic Loss metric
        Parameters
        ----------
        y_true : array, shape = [n_samples]
        true class, intergers in [0, n_classes - 1)
        y_pred : array, shape = [n_samples, n_classes]
        Returns
        -------
        loss : float
        """
        
        # copy model to mimic pass-by value
        #model = deepcopy(estimator)
        model = estimator

        # check to see if model is a pipeline object or not
        if isinstance(model, sklearn.pipeline.Pipeline):
            data_type = type(model._final_estimator)
        else:
            data_type = type(model)

        name = filter(str.isalnum, str(data_type).split(".")[-1])

        # check to see if model file exist
        #if not os.path.isfile('models/'+name+'.pkl'):
        model.fit(X_train, y_train)
        #joblib.dump(model, 'models/'+name+'.pkl')
        #else:
        #    model = joblib.load('models/'+name+'.pkl')

        # use subplot to extract axis to add ks and p-value to plot
        self.fig, self.ax = plt.subplots(figsize=(8, 6))

        # Customize the major grid
        self.ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
        self.ax.set_facecolor('white')

        if not hasattr(model, 'predict_proba'): # use decision function
            d = model.decision_function(sp.sparse.vstack([X_train, X_test]))
            bin_edges_low_high = np.linspace(min(d), max(d), bins + 1)
        else: # use prediction function
            bin_edges_low_high = np.linspace(0., 1., bins + 1)

        label_name = ""
        y_scores = []
        for X, y in [(X_train, y_train), (X_test, y_test)]:

            if hasattr(model, 'predict_proba'):
                label_name = 'Prediction Probability'
                y_scores.append(model.predict_proba(X[y > 0])[:, 1])
                y_scores.append(model.predict_proba(X[y < 1])[:, 1])
            else:
                label_name = 'Decision Function'
                y_scores.append(model.decision_function(X[y > 0]))
                y_scores.append(model.decision_function(X[y < 1]))
            
        width = np.diff(bin_edges_low_high)

        # Signal training histogram
        #np.set_printoptions(threshold='nan')
        hist_sig_train, bin_edges = np.histogram(y_scores[0], bins=bin_edges_low_high)
        hist_sig_train = hist_sig_train / np.sum(hist_sig_train, dtype=np.float32)
    
        plt.bar(bin_edges[:-1], hist_sig_train, width=width, color='r', alpha=0.5,
                label='signal (train)')

        # Background training histogram
        hist_bkg_train, bin_edges = np.histogram(y_scores[1], bins=bin_edges_low_high)

        hist_bkg_train = hist_bkg_train / np.sum(hist_bkg_train, dtype=np.float32)

        plt.bar(bin_edges[:-1], hist_bkg_train, width=width,
                color='steelblue', alpha=0.5, label='background (train)')

        # Signal test histogram
        hist_sig_test, bin_edges = np.histogram(y_scores[2], bins=bin_edges_low_high)

        hist_sig_test = hist_sig_test / np.sum(hist_sig_test, dtype=np.float32)
        scale = len(y_scores[2]) / np.sum(hist_sig_test, dtype=np.float32)
        err = np.sqrt(hist_sig_test * scale) / scale

        plt.errorbar(bin_edges[:-1], hist_sig_test, yerr=err, fmt='o', c='r', label='signal (test)')

        # Background test histogram
        hist_bkg_test, bin_edges = np.histogram(y_scores[3], bins=bin_edges_low_high)

        hist_bkg_test = hist_bkg_test / np.sum(hist_bkg_test, dtype=np.float32)
        scale = len(y_scores[3]) / np.sum(hist_bkg_test, dtype=np.float32)
        err = np.sqrt(hist_bkg_test * scale) / scale

        plt.errorbar(bin_edges[:-1], hist_bkg_test, yerr=err, fmt='o', c='steelblue', #range=low_high,
                     label='background (test)')

        # Estimate ks-test and p-values as an indicator of overtraining of fit model
        s_ks, s_pv = ks_2samp(hist_sig_test, hist_sig_train)
        b_ks, b_pv = ks_2samp(hist_bkg_test, hist_bkg_train)
        
        #s_ks, s_pv = ks_weighted_2samp(y_scores[0], y_scores[2],
        #                               signal_sample_weight_train, signal_sample_weight_test)
        #b_ks, b_pv = ks_weighted_2samp(y_scores[1], y_scores[3],
        #                               background_sample_weight_train,
        #                               background_sample_weight_test)

        
        self.ax.set_title("%s: sig (bkg)\nks: %0.3f (%0.3f)\np-value: %0.3f (%0.3f)"
                          % (name, s_ks, b_ks, s_pv, b_pv), fontsize=14)

        plt.xlabel(label_name)
        plt.ylabel('Arbitrary units')
        
        leg = plt.legend(loc='best', frameon=False, fancybox=False, fontsize=12)
        leg.get_frame().set_edgecolor('w')

        frame = leg.get_frame()
        frame.set_facecolor('White')

        return plt.show()


    ## Define validation plots
    def validation_curve(self, estimators, X_train, y_train, param_name,
                         params, param_range, cv=3,
                         scoring="neg_log_loss", logx=False,
                         n_jobs=1):
        """
        Draw histogram of the DataFrame's series comparing the distribution
        in `signal` to `background`.

        Parameters
        ----------
        models : dictionary, shape = [n_models]
        X : DataFrame, shape = [n_samples, n_classes]
        y : DataFrame, shape = [n_classes]
        param_range :

        param_name :

        cv :
        scoring :
        n_jobs :

        Returns
        -------
        plot : matplotlib plot
        """

        """
        Describe possible kwargs values

        Keys
        ----------
        """

        # deep compy of model(s)
        models = deepcopy(estimators)

        # line width
        lw = 2

        # check to see if models is a list
        if not isinstance(models, list):
            models = [models]
        # check to see if model is a pipeline object or not
        if isinstance(models[0], sklearn.pipeline.Pipeline):
            data_type = type(models[0]._final_estimator)
        else:
            data_type = type(models[0])

        # plot title
        name = filter(str.isalnum, str(data_type).split(".")[-1])
        title = "Validation Curves (%s)" % name

        # create blank canvas
        self.fig, self.ax = plt.subplots(figsize=(8, 6))

        self.ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
        #self.ax.set_facecolor('white')

        for n, model in enumerate(models):
            # validation scores
            train_scores, test_scores = validation_curve(
                model, X_train, y_train,
                param_name=param_name,
                param_range=param_range, cv=cv,
                scoring=scoring, n_jobs=n_jobs)

            # mean train scores
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std  = np.std(train_scores, axis=1)

            # mean test scores
            test_scores_mean  = np.mean(test_scores, axis=1)
            test_scores_std   = np.std(test_scores, axis=1)

            # extract information for legend
            label = 'placeholder ('

            #for i, p_name in enumerate(params.keys()):
            #    param = model.get_params(deep=True)[p_name]
            #    if i != len(params.keys())-1:
            #        label = label+p_name.replace(name.lower()+'__','')+'=%.1f, ' % param
            #    else:
            #        label = label+p_name.replace(name.lower()+'__','')+'=%.1f' % param
            label=label+')'

            # plot validation curve
            if logx is True:
                plt.semilogx(param_range, train_scores_mean, '--',
                             label=label.replace('placeholder','Training'),
                             color="darkorange", lw=lw)

                plt.semilogx(param_range, test_scores_mean,
                             label=label.replace('placeholder','Test'),
                             color="navy", lw=lw)
            else:
                plt.plot(param_range, train_scores_mean, '--',
                         label=label.replace('placeholder','Training'),
                         color="darkorange", lw=lw)

                plt.plot(param_range, test_scores_mean,
                         label=label.replace('placeholder','Test'),
                         color="navy", lw=lw)

            plt.fill_between(param_range, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.2,
                             color="darkorange", lw=lw)

            plt.fill_between(param_range, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.2,
                             color="navy", lw=lw)

            #colour = test_line[-1].get_color()
            #plt.axvline(x=best_iter, color=colour)

        # plot title
        plt.title(title, fontsize=14)

        plt.xlabel(str(param_name).replace(name.lower()+'__',''))
        plt.ylabel(scoring)

        # x-axis range
        plt.xlim([min(param_range), max(param_range)])

        plt.legend(loc='best', frameon=False, fancybox=True, fontsize=12)

        return plt.show()


    ## Define ROC curve plot
    def roc_curve(self, estimators, X_train, X_test, y_train, y_test):
        """Roc curve metric plotter.
        
        Parameters
        ----------
        models : dictionary, shape = [n_models]
        X : DataFrame, shape = [n_samples, n_classes]
        y : DataFrame, shape = [n_classes]

        Returns
        -------
        roc : matplotlib plot
        """

        #models = deepcopy(estimators)
        models = estimators

        # contains rates for ML classifiers
        fpr = {}
        tpr = {}
        roc_auc = {}
    
        # Customize the major grid
        self.fig, self.ax = plt.subplots()
        self.ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
        self.ax.patch.set_facecolor('white')
    
        # Include random by chance 'luck' curve
        plt.plot([0, 1], [0, 1], '--', color=(0.1, 0.1, 0.1), label='Luck')
        
        # Loop through classifiers
        for (name, model) in models.items():
        
            print("\n\x1b[1;31mBuilding model ...\x1b[0m")
            model.fit(X_train, y_train)

            y_predicted = model.predict(X_test)
        
            if hasattr(model, "predict_proba"):
                decisions = model.predict_proba(X_test)[:, 1]
            else:  # use decision function
                decisions = model.decision_function(X_test)
        
            print("\tArea under ROC curve for %s: %.4f"%(name, roc_auc_score(y_test,decisions)))
        
            #print classification_report(y_test, y_predicted, target_names=['signal', 'background'])
            print("\tScore of test dataset: {:.5f}".format(model.score(X_test, y_test)))

            # Calculate the area under the ROC curve
            fpr[name], tpr[name], thresholds = roc_curve(y_test, decisions)

            # Configure AUC ROC per ML classifiers
            roc_auc[name] = auc(fpr[name], tpr[name])
    
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 
                        'green', 'yellow', 'SlateBlue', 'DarkSlateGrey',
                        'CadetBlue', 'Chocolate', 'darkred', 'GoldenRod'])
  
        for (name, model), color in zip(models.items(), colors):

            plt.plot(fpr[name], tpr[name], # physics version: tpr[name], 1-fpr[name], 
                     color=color, lw=2,
                     label='%s (AUC = %0.3f)'%(name, roc_auc[name]))                 
    
        # Plot all ROC curves
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("Receiver operating characteristic ({} events)".format(X_train.shape[0]+X_test.shape[0]))
        leg = plt.legend(loc="lower right", frameon=True, fancybox=True, fontsize=8) # loc='best'
        leg.get_frame().set_edgecolor('w')
        frame = leg.get_frame()
        frame.set_facecolor('White')
    
        return plt.show()


    ## Define Cross-validated ROC curve plot
    def roc_curve_cv(self, estimators, X, y, folds_included=False):
        """Cross-validated Roc curve metric plotter.

        Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html

        Parameters
        ----------
        models : dictionary, shape = [n_models]
        X : DataFrame, shape = [n_samples, n_classes]
        y : DataFrame, shape = [n_classes]

        Returns
        -------
        roc : matplotlib plot
        """

        # Classification and ROC analysis
        classifier = estimators

        # Run classifier with cross-validation and plot ROC curves
        n_splits=5
        cv = StratifiedKFold(n_splits=n_splits)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        # Customize the major grid
        self.fig, self.ax = plt.subplots()
        self.ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
        self.ax.patch.set_facecolor('white')

        i = 0
        for train, test in cv.split(X, y):
            probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            if folds_included == True:
                plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
            i += 1
    
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Luck', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("Receiver operating characteristic ({} events)".format(X.shape[0]))
        leg = plt.legend(loc="lower right", frameon=True, fancybox=True, fontsize=8) # loc='best'
        leg.get_frame().set_edgecolor('w')
        frame = leg.get_frame()
        frame.set_facecolor('White')    

        return plt.show()



    ## Learning curve
    def learning_curve(self, model, X_train, y_train,
                       ylim=None, cv=None, n_jobs=1,
                       train_sizes=np.linspace(0.1, 1.0, 10, endpoint=True)):
        """
        Generate a simple plot of the test and training learning curve.

        Parameters
        ----------
        model : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        X_train : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y_train : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

        cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

          For integer/None inputs, if ``y`` is binary or multiclass,
          :class:`StratifiedKFold` used. If the estimator is not a classifier
          or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

          Refer :ref:`User Guide <cross_validation>` for the various
          cross-validators that can be used here.

          n_jobs : integer, optional
             Number of jobs to run in parallel (default 1).

          train_sizes = np.linspace(0.1, 1.0, 10, endpoint=True) produces
             8 evenly spaced points in the range 0 to 10
          """

        # check to see if model is a pipeline object or not
        if isinstance(model, sklearn.pipeline.Pipeline):
            data_type = type(model._final_estimator)
        else:
            data_type = type(model)

        # plot title
        name = filter(str.isalnum, str(data_type).split(".")[-1])
        title = "Learning Curves (%s)" % name

        # create blank canvas
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
        self.ax.set_facecolor('white')

        train_sizes_abs, train_scores, test_scores = learning_curve(model,
                                                                    X_train, y_train,
                                                                    train_sizes=np.linspace(0.1, 1.0, 10),
                                                                    cv=cv,
                                                                    scoring=None,
                                                                    exploit_incremental_learning=False,
                                                                    n_jobs=n_jobs,
                                                                    pre_dispatch="all",
                                                                    verbose=0)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std  = np.std(train_scores, axis=1)

        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std  = np.std(test_scores, axis=1)

        # plot the std deviation as a transparent range at each training set size
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")

        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")

        # plot the average training and test score lines at each training set size
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")

        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.title(title, fontsize=14)

        # sizes the window for readability and displays the plot
        # shows error from 0 to 1.1
        if ylim is not None:
            plt.ylim(*ylim)
            #plt.ylim(-.1, 1.1)

        plt.xlabel("Training set size")
        plt.ylabel("Score")

        leg = plt.legend(loc="best", frameon=True, fancybox=False, fontsize=12)
        leg.get_frame().set_edgecolor('w')

        frame = leg.get_frame()
        frame.set_facecolor('White')

        # box-like grid
        #plt.grid(figsize=(8, 6))

        #plt.gca().invert_yaxis()

        return plt.show()


    ## Plot signal and background distributions for some variables
    # The first two arguments select what is 'signal'
    # and what is 'background'. This means you can
    # use it for more general comparisons of two
    # subsets as well.
    def signal_background(self, data1, data2, column=None, grid=True,
                          xlabelsize=None, xrot=None, ylabelsize=None,
                          yrot=None, ax=None, sharex=False,
                          sharey=False, figsize=None,
                          layout=None, bins=10, sample_weights='globalTimesEventWeight',
                          **kwds):
        """Draw histogram of the DataFrame's series comparing the distribution
        in `data1` to `data2`.

        data1: DataFrame
        data2: DataFrame
        column: string or sequence
           If passed, will be used to limit data to a subset of columns
        grid : boolean, default True
           Whether to show axis grid lines
        xlabelsize : int, default None
           If specified changes the x-axis label size
        xrot : float, default None
           rotation of x axis labels
        ylabelsize : int, default None
           If specified changes the y-axis label size
        yrot : float, default None
           rotation of y axis labels
        ax : matplotlib axes object, default None
        sharex : bool, if True, the X axis will be shared amongst all subplots.
        sharey : bool, if True, the Y axis will be shared amongst all subplots.
        figsize : tuple
           The size of the figure to create in inches by default
        layout: (optional) a tuple (rows, columns) for the layout of the histograms
        bins: integer, default 10
           Number of histogram bins to be used
        kwds : other plotting keyword arguments
           To be passed to hist function
        """
        manification=20

        # NOTE: All sample weights for both signal and background are set to 1 for now
        background_weight =  pd.DataFrame(1., index=np.arange(data1.shape[0]), columns=['sample_weights']) 
        signal_weight = pd.DataFrame(1., index=np.arange(data2.shape[0]), columns=['sample_weights'])*manification

        if "alpha" not in kwds:
            kwds["alpha"] = 0.5

        w, h = (10, 6)
        figsize = (w, h)

        if column is not None:
            if not isinstance(column, (list, np.ndarray, Index)):
                column = [column]
                data1 = data1[column]
                data2 = data2[column]

        data1 = data1._get_numeric_data()
        data2 = data2._get_numeric_data()
        naxes = len(data1.columns)

        self.fig, self.axes = plt.subplots(nrows=4, ncols=4,
                                           squeeze=False,figsize=figsize)

        #xs = plotting._flatten(axes)
        self.xs = self.axes.flat

        for i, col in enumerate(com._try_sort(data1.columns)):
            self.ax = self.xs[i]
            low = min(data1[col].min(), data2[col].min())
            high = max(data1[col].max(), data2[col].max())
            self.ax.hist(data1[col].dropna().values, weights=background_weight,
                         bins=bins, histtype='stepfilled', range=(low,high), **kwds)
            self.ax.hist(data2[col].dropna().values, weights=signal_weight,
                         bins=bins, histtype='stepfilled', range=(low,high), **kwds)
            self.ax.set_title(col)
            self.ax.legend(['background', 'signal (%s)'% (manification)], loc='best')
            self.ax.set_facecolor('white')

            # Customize the major grid
            self.ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
            self.ax.set_facecolor('white')

        #plotting._set_ticks_props(axes, xlabelsize=xlabelsize, xrot=xrot,
        #                         ylabelsize=ylabelsize, yrot=yrot)
        self.fig.subplots_adjust(wspace=0.5, hspace=0.8)

        return plt.show()


    ## Heat map of the first layer weights in a neural network
    def dnn_weight_map(self, model, X_train, y_train, features):

        #estimator = deepcopy(model)
        estimator = model

        SMALL_SIZE  = 5
        MEDIUM_SIZE = 8
        BIGGER_SIZE = 10

        plt.rc('font',  size=SMALL_SIZE)        # controls default text sizes
        plt.rc('axes',  labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

        # train model
        estimator.fit(X_train, y_train)

        W, b = estimator.model.layers[0].get_weights()

        W = np.squeeze(W)
        W = np.round(W, decimals=1)

        # The normal figure
        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(1,1,1)
        plt.imshow(W, interpolation='nearest', cmap='viridis', origin='lower')

        # Limits for the extent
        x_start = -1.0
        x_end   = W.shape[1]-1
        y_start = -1.0
        y_end   = len(features)-1

        # Add the text
        jump_x = 0.9 #1.
        jump_y = 0.9 #1.

        x_positions = np.linspace(start=x_start, stop=x_end, num=W.shape[1], endpoint=False)
        y_positions = np.linspace(start=y_start, stop=y_end, num=(len(features)-1), endpoint=False)

        for y_index, y in enumerate(y_positions):
            for x_index, x in enumerate(x_positions):
                label = W[y_index, x_index]
                text_x = x + jump_x
                text_y = y + jump_y
                ax.text(text_x, text_y, label, color='black', ha='center', va='center')
        

        # Heat map
        plt.title("Neural network first layer weights")
        plt.yticks(range(len(features)), features)
        plt.xticks(range(W.shape[1]), [i for i in xrange(W.shape[1])])
        plt.xlabel("Columns in weight matrix")
        plt.ylabel("Input feature")
        plt.colorbar()
        plt.grid("off")
        fig.tight_layout()
        self.fig = fig

        return plt.show()

    ## Scatter plots between variables
    def scatter_plot(self, data, sample_weight='globalTimesEventWeight'):

        sample_weights = data[sample_weight].values

        sns.set(style="ticks", color_codes=True)
        
        g = sns.PairGrid(data.drop([sample_weight], axis=1),
                         hue="y", palette="GnBu_d",
                         hue_kws={"marker": ["o", "s"]})
        g.map_diag(plt.hist)
        g.map_offdiag(plt.scatter, s=sample_weights, alpha=0.7, edgecolor="white")
        g.add_legend();

        xlabels, ylabels = [],[]
        
        for ax in g.axes[-1,:]:
            xlabel = ax.xaxis.get_label_text()
            xlabels.append(xlabel)
        for ax in g.axes[:,0]:
            ylabel = ax.yaxis.get_label_text()
            ylabels.append(ylabel)

        for i in range(len(xlabels)):
            for j in range(len(ylabels)):
                g.axes[j,i].xaxis.set_label_text(xlabels[i])
                g.axes[j,i].yaxis.set_label_text(ylabels[j])

        return plt.show()


    ## Define calibration curve (reliability curve)
    def calibration_curve(self, estimators, X, y, 
                          fig_index=2, n_bins=10, cv=2,
                          sample_weight='globalTimesEventWeight'):
        """Plot calibration curve for est w/o and with calibration. """
        
        #est = deepcopy(estimators)
        est = estimators
        #est = clone(estimators)

        # Split development set into a train and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                            random_state=42)

        # Extract sample weights
        #X_train = X_train.drop(sample_weight, axis=1, inplace=False)
        #X_test = X_test.drop(sample_weight, axis=1, inplace=False)

        # Calibrated with isotonic calibration
        isotonic = CalibratedClassifierCV(est,
                                          cv=cv,
                                          method='isotonic')

        # Calibrated with sigmoid calibration
        sigmoid = CalibratedClassifierCV(est,
                                         cv=cv,
                                         method='sigmoid')

        self.fig = plt.figure(fig_index, figsize=(6, 6))
        self.ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        self.ax2 = plt.subplot2grid((3, 1), (2, 0))

        self.ax1.plot([0, 1], [0, 1], "--", label="Perfectly calibrated")

        # Order of classifier being fitted maters, est classifier must be last not sure why yet
        for clf, name in [(isotonic, est.steps[1][0] + '_Isotonic'),
                          (sigmoid, est.steps[1][0] + '_Sigmoid'),
                          (est, est.steps[1][0])]: # Also called Platt Scaling

            #clf.fit(X_train, y_train, sample_weight
            if  clf.__class__.__name__ == "CalibratedClassifierCV":
                clf.fit(X_train,y_train)
            else:
                clf.fit(X_train,y_train)

            y_pred = clf.predict(X_test)

            if hasattr(clf, "predict_proba"):
                prob_pos = clf.predict_proba(X_test)[:, 1]
            else:  # use decision function
                prob_pos = clf.decision_function(X_test);
                prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

            clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
            print("\n\x1b[1;31mclassifier %s:\x1b[0m" % name)
            print("\tBrier: %1.3f" % (clf_score))
            print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
            print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
            print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

            fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

            self.ax1.plot(mean_predicted_value, fraction_of_positives, "o-",
                          label="%s (%1.3f)" % (name, clf_score))

            self.ax2.hist(prob_pos, range=(0, 1), bins=n_bins, label=name,
                          histtype="step", lw=2)

        self.ax1.set_ylabel("Fraction of positives")
        self.ax1.set_ylim([-0.05, 1.05])
        self.ax1.legend(loc="lower right")
        self.ax1.set_title('Calibration plots  (reliability curve)')

        # Customize the major grid
        self.ax1.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
        self.ax1.set_facecolor('white')

        self.ax2.set_xlabel("Mean predicted value")
        self.ax2.set_ylabel("Count")
        self.ax2.legend(loc="best", ncol=1)

        # Customize the major grid
        self.ax2.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
        self.ax2.set_facecolor('white')

        plt.tight_layout()

        return plt.show()
      
      ## Save plot
    def savePlots(self, dir='plots', filename=''):
        """ 
        Save plot to format e.g. pdf
        """
        # checks if directory exists and if not creates it 
        self.ensure_dir(dir)
        
        # save file
        self.fig.savefig(dir+'/'+filename)

        return self.fig

    ## Check folder exist
    def ensure_dir(self, directory):
        """
        When directory is not present, create it.
        Arguments: 
        directory: name of directory.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
