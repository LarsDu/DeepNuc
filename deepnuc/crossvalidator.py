import os
import numpy as np
from nucbinaryclassifier import NucBinaryClassifier
from nucregressor import NucRegressor
import nucconvmodel
from databatcher import DataBatcher
import tensorflow as tf
from collections import defaultdict


#from logger import Logger
#import sys

class CrossValidator(object):
    def __init__(self,
                 params,
                 nuc_data,
                 cv_save_dir,
                 seed,
                 k_folds=3,
                 test_frac=0.15):


        
        """
        Perform k-fold cross validation on a dataset. nuc_data will be divided into
        k training and testing sets.
        :param params: A ModelParams object. Needed for passing params to NucBinaryClassifier
        :param nuc_data: An object derived from BaseNucData
        :param cv_save_dir: Save k models under this directory
        :param seed: An integer used to seed random shuffling 
        :param nn_method_key: string name of method from nucconvmodels.py
        :param k_folds: Number of k-folds 
        :param test_frac: Fraction of nuc_data to be used for cross validation 

        """

        self.params = params
        self.nuc_data = nuc_data

        # NucBinaryClassifier has this static flag set to True
        # NucRegressor has this flag set to False 
        self.use_onehot_labels=True
        
        try:
            os.makedirs(cv_save_dir)
        except OSError:
            if not os.path.isdir(cv_save_dir):
                raise

        self.cv_save_dir = cv_save_dir

        self.k_folds = k_folds

        self.test_frac = test_frac

        #Make sure k_folds values make sense
        if (self.k_folds>0) and (1./self.k_folds < self.test_frac):
            print "ERROR!!"
            print('test_frac ',self.test_frac,' too large for k=',
            self.k_folds,' fold.') 

        self.nn_method = nucconvmodel.methods_dict[params.inference_method_key]

        self.train_size = int((1-self.test_frac)*self.nuc_data.num_records)
        self.test_size = int(self.nuc_data.num_records-self.train_size)

        #The indices of the following lists correspond to different k-folds
        self.train_batcher_list = [None]*self.k_folds
        self.test_batcher_list = [None]*self.k_folds
        self.nuc_classifier_list = [None]*self.k_folds

        self.train_results=[{}]*self.k_folds
        self.test_results=[{}]*self.k_folds

        self.optim_train_results=[{}]*self.k_folds
        self.optim_test_results=[{}]*self.k_folds
        """
        In order to save training test splits, the shuffled indices are generated
        from a seed value. Saving the seed value,k_folds, and test_frac will
        allow recovery of the exact same cross validation training/test splits
        """
        self.seed = seed

        self.shuffled_indices = np.random.RandomState(self.seed).\
                                            permutation(range(self.nuc_data.num_records))


            

    def run(self):
        for k in range(self.k_folds):
            test_indices = self.shuffled_indices[(self.test_size*k):(self.test_size*(k+1))]
            train_indices = np.setdiff1d(self.shuffled_indices, test_indices)

            self.train_batcher_list[k] = DataBatcher(self.nuc_data,
                                                     train_indices,
                                                     self.use_onehot_labels)



            self.test_batcher_list[k] = DataBatcher(self.nuc_data,
                                                    test_indices,
                                                    self.use_onehot_labels)


            print "\n\n\nStarting k_fold", str(k)
            #with tf.variable_scope("kfold_"+str(k)):

            kfold_dir = self.cv_save_dir+os.sep+'kfold_'+str(k)
            with tf.Session() as sess:
                self.nuc_classifier_list[k] = NucBinaryClassifier( sess,
                                                                 self.train_batcher_list[k],
                                                                 self.test_batcher_list[k],
                                                                 self.params.num_epochs,
                                                                 self.params.learning_rate,
                                                                 self.params.batch_size,
                                                                 self.params.seq_len,
                                                                 kfold_dir,
                                                                 self.params.keep_prob,
                                                                 self.params.beta1,
                                                                 self.params.inference_method_key
                                                                 )
                print "Building model for k-fold", k
                self.nuc_classifier_list[k].build_model()

                print "Training k-fold", k
                print "Validation training set size",self.train_size
                print "Validation testing set size",self.test_size
                #note: train() will also load saved models
                self.train_results[k],self.test_results[k] = self.nuc_classifier_list[k].train()
                self.optim_train_results[k] = self.nuc_classifier_list[k].get_optimal_metrics(
                                               self.nuc_classifier_list[k].train_metrics_vector,
                                                                   metric_key="auroc")
                self.optim_test_results[k]  = self.nuc_classifier_list[k].get_optimal_metrics(
                                                  self.nuc_classifier_list[k].test_metrics_vector,
                                                                   metric_key="auroc")

                plot_label = "k-fold "+str(k)
                self.nuc_classifier_list[k].plot_test_epoch_vs_metric('auroc',
                                                                      plot_label,
                                                                      save_plot=True)
                self.nuc_classifier_list[k].plot_test_epoch_vs_metric('auroc',
                                                                      plot_label,
                                                                      save_plot=True)
                                                                 
            tf.reset_default_graph()

        
        #Print cross validation results
        self.print_results()

    

      
                    
    def print_results(self):
        print "Cross validation parameters:"
        print "Training set size:\t",self.train_size
        print "Testing set size:\t",self.test_size
        print "Seed value", self.seed
        print "k-folds", self.k_folds
        print "test_frac",self.test_frac
        for k in range(self.k_folds):
            print "\nTRAINING metrics for k-fold",k
            self.nuc_classifier_list[k].print_metrics(self.train_results[k])
        for k in range(self.k_folds):
            print "\nTESTING metrics for k-fold",k
            self.nuc_classifier_list[k].print_metrics(self.test_results[k])

            
        print "\n\n"
        self.print_avg_k_metrics()
        


    def print_avg_k_metrics(self):
        avg_train_results,avg_test_results = self.calc_avg_k_metrics()
        
        for result,message in ([(avg_train_results,'TRAIN'),(avg_test_results,'TEST')]):
            print 'AVERAGE {} metrics across {}-fold cross validation'.\
                                                         format(message,self.k_folds)
            for k,v in result.items():
                if type(v) != np.ndarray:
                    print '\t',k,":\t",v
            print "\n"
    
                    
    def calc_avg_k_metrics(self):
        avg_train_results = defaultdict(float)
        avg_test_results = defaultdict(float)
        #train_results is list
        #train_results[k] is dict
        in_results = [self.train_results,self.test_results] 
        out_results = [avg_train_results,avg_test_results] #avg_train_results is dict
        for i in range(2):
            for k_result in in_results[i]:
                for metric,val in k_result.viewitems():
                    if type(val) != np.ndarray and metric != 'epoch' and metric != 'step':
                        out_results[i][metric] += val
            for key in out_results[i].viewkeys():
                #Get the average of each metric across k_folds
                out_results[i][key] /= float(self.k_folds)
        return avg_train_results,avg_test_results




    
    '''
    def best_precision_classifier(self):
        """Return the classifier with the best precision"""
        pass
        

    def best_recall_classifier(self):
        """Return the classifier with the best recall"""
        pass

    
    def best_auc_classifier(self):
        """Return the classifier with the best auc"""
        pass
    '''


class RegressorCrossValidator(CrossValidator):
    def __init__(self,
                 params,
                 nuc_data,
                 cv_save_dir,
                 seed,
                 k_folds=3,
                 test_frac=0.15,
                 classification_threshold=None,
                 output_scale=[0,1]):

        super(RegressorCrossValidator, self).__init__(
                                             params=params,
                                             nuc_data=nuc_data,
                                             cv_save_dir=cv_save_dir,
                                             seed=seed,
                                             k_folds=k_folds,
                                             test_frac=test_frac)
                                             


        self.classification_threshold = classification_threshold
        self.use_onehot_labels=False
        self.output_scale = output_scale
        



    def run(self):
        for k in range(self.k_folds):
            test_indices = self.shuffled_indices[(self.test_size*k):(self.test_size*(k+1))]
            train_indices = np.setdiff1d(self.shuffled_indices, test_indices)#These are now sorted

            self.train_batcher_list[k] = DataBatcher(self.nuc_data,
                                                     train_indices,
                                                     self.use_onehot_labels)
            self.test_batcher_list[k] = DataBatcher(self.nuc_data,
                                                    test_indices,
                                                    self.use_onehot_labels)

            print "\n\n\nStarting k_fold", str(k)
            #with tf.variable_scope("kfold_"+str(k)):

            kfold_dir = self.cv_save_dir+os.sep+'kfold_'+str(k)
           
            kfold_dir = self.cv_save_dir+os.sep+'kfold_'+str(k)
            with tf.Session() as sess:
                self.nuc_classifier_list[k] = NucRegressor( sess,
                                                                 self.train_batcher_list[k],
                                                                 self.test_batcher_list[k],
                                                                 self.params.num_epochs,
                                                                 self.params.learning_rate,
                                                                 self.params.batch_size,
                                                                 self.params.seq_len,
                                                                 kfold_dir,
                                                                 self.params.keep_prob,
                                                                 self.params.beta1,
                                                                 self.params.concat_revcom_input,
                                                                 self.classification_threshold,
                                                                 self.output_scale)
                print "Building model for k-fold", k
                self.nuc_classifier_list[k].build_model()

                print "Training k-fold", k
                print "Validation training set size",self.train_size
                print "Validation testing set size",self.test_size
                #note: train() will also load saved models
                self.train_results[k],self.test_results[k] = self.nuc_classifier_list[k].train()
            tf.reset_default_graph()

        
        #Print cross validation results
        self.print_results()



            
