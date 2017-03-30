import os
import numpy as np
from nucclassifier import NucClassifier
import nucconvmodel
from databatcher import DataBatcher
import tensorflow as tf


#from logger import Logger
#import sys

class CrossValidator:
    def __init__(self,
                 sess,
                 params,
                 nuc_data,
                 cv_save_dir,
                 nn_method=nucconvmodel.inferenceA,
                 k_folds=None,
                 test_frac=None):
                
        """
        Perform k-fold cross validation on a dataset. nuc_data will be divided into
        k training and testing sets.

        :param nuc_data: An object derived from BaseNucData
        :param save_dir: Save k models under this directory
        :param params: A ModelParams object. Needed for passing params to NucClassifier
        :param nn_method: A dtlayers.py constructed neural network (examples in nucconvmodel.py)
        :param k_folds: Number of k-folds 
        :param test_frac: Fraction of nuc_data to be used for cross validation 

        """

        self.sess = sess
        self.params = params
        self.nuc_data = nuc_data
        self.cv_save_dir = cv_save_dir
        
        if k_folds == None:
            print "Number of k folds not specified. Setting to 3"
            self.k_folds = 3
        else:
            self.k_folds = k_folds

        if test_frac == None:
            print "Test fraction of k-folds not specified. Setting to 0.15"
            self.test_frac = 0.15
        else:
            self.test_frac = test_frac

        #Make sure k_folds values make sense
        if (self.k_folds>0) and (1./self.k_folds < self.test_frac):
            print "ERROR!!"
            print('test_frac ',self.test_frac,' too large for k=',
            self.k_folds,' fold.') 

            

        self.nn_method = nn_method

        self.train_size = int((1-self.test_frac)*self.nuc_data.num_records)
        self.test_size = int(self.nuc_data.num_records-self.train_size)

        #The indices of the following lists correspond to different k-folds
        self.train_batcher_list = [None]*self.k_folds
        self.test_batcher_list = [None]*self.k_folds
        self.nuc_classifier_list = [None]*self.k_folds

        self.train_results=[{}]*self.k_folds
        self.test_results=[{}]*self.k_folds
        
        """
        In order to save training test splits, the shuffled indices are generated
        from a seed value. Saving the seed value,k_folds, and test_frac will
        allow recovery of the exact same cross validation training/test splits
        """
        if seed == None:
            self.seed = np.random.randint(1e7)
            print "Setting shuffling seed for cross validation to",self.seed
        else:
            self.seed = seed
        
        #self.seed = np.random.randint(1e7)
        self.shuffled_indices = np.random.RandomState(self.seed).\
                                            permutation(range(self.nuc_data.num_records))


        
        for k in range(self.k_folds):
            test_indices = self.shuffled_indices[(self.test_size*k):(self.train_size*(k+1))]
            train_indices = np.setdiff1d(self.shuffled_indices, test_indices)
            self.train_batcher_list[k] = DataBatcher(self.nuc_data, train_indices)
            self.test_batcher_list[k] = DataBatcher(self.nuc_data, test_indices)

            print "\n\n\nStarting k_fold", str(k)
            
            
            kfold_dir = self.cv_save_dir+os.sep+'kfold_'+str(k)

            with tf.variable_scope("kfold_"+str(k)):
            
                self.nuc_classifier_list[k] = NucClassifier( self.sess,
                                                             self.train_batcher_list[k],
                                                             self.test_batcher_list[k],
                                                             self.params.num_epochs,
                                                             self.params.learning_rate,
                                                             self.params.batch_size,
                                                             self.params.seq_len,
                                                             kfold_dir,
                                                             self.params.keep_prob,
                                                             self.params.beta1)
                print "Building model for k-fold", k
                self.nuc_classifier_list[k].build_model(self.nn_method)

                print "Training k-fold", k
                #note: train() will also load saved models
                self.train_results[k],self.test_results[k] = self.nuc_classifier_list[k].train()
                #tf.reset_default_graph()
                                                       
        
        self.mean_train_precision = np.mean([d['precision'] for d in self.train_results])
        self.mean_test_precision = np.mean([d['precision'] for d in self.test_results])

        #Print cross validation results
        self.print_results()

        
                    
    def print_results(self):
        print "Training set size:\t",self.train_size
        print "Testing set size:\t",self.test_size
        print "Mean training precision", self.mean_train_precision
        print "Mean testing precision", self.mean_test_precision
        print "Seed value", self.seed
        print "k-folds", self.k_folds
        print "test_frac",self.test_frac
        for k in range(self.k_folds):
            print "\nTraining metrics for k-fold",k
            self.print_metrics(self.train_results[k])
            print "\nTesting metrics for k-fold",k
            self.print_metrics(self.test_results[k])



            
    def print_metrics(self,metrics_dict):
        print "auROC:\t",metrics_dict["auroc"]
        print "auPRC:\t",metrics_dict["auprc"]
        print "Accuracy:\t",metrics_dict["accuracy"]
        print "Precision:\t",metrics_dict["precision"]
        print "Recall:\t",metrics_dict["recall"]
        print "F1-score:\t",metrics_dict["f1_score"]
        print "Support:\t",metrics_dict["support"]
              
    def best_precision_classifier(self):
        """Return the classifier with the best precision"""
        pass
        

    def best_recall_classifier(self):
        """Return the classifier with the best recall"""
        pass

    
    def best_auc_classifier(self):
        """Return the classifier with the best auc"""
        pass

