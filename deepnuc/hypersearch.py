#import tensorflow as tf
from modelparams import *
from crossvalidator import CrossValidator
import sys
import os
from logger import Logger
from databatcher import DataBatcher 
from nucbinaryclassifier import NucBinaryClassifier
from nucregressor import NucRegressor
import pickle


class GridSearch(object):
    def __init__(self,
                 grid_params_obj,
                 nuc_data,
                 gs_save_dir,
                 seed = 1212516,
                 k_folds = 3,
                 test_frac =0.20,
                 fig_title_prefix=''):
        
        self.nuc_data = nuc_data
        self.grid_params = grid_params_obj
        self.grid_params_list = grid_params_obj.grid_params_list
        self.save_dir = gs_save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.best_model_dir = self.save_dir+os.sep+'best_model'
        if not os.path.exists(self.best_model_dir):
            os.makedirs(self.best_model_dir)

        self.fig_title_prefix = fig_title_prefix
        
        self.seed = seed #For train/test cross validation splits
        self.k_folds =k_folds
        self.test_frac = test_frac




        #Params for saving certain metrics
        self.cv_save_dirs = [ self.save_dir+os.sep+'gparam_cv{}'.format(i) \
                              for i in range(len(self.grid_params_list))]

        self.grid_data_file = self.save_dir+os.sep+'gridsearch_data.p'

        #These values need to be filled with either run() or load()
        self.cv_test_results = []
        self.best_model_params = None

           
                        
    def run(self):

        for i,gparam in enumerate(self.grid_params_list):

            self.log_fname = self.cv_save_dirs[i]+os.sep+"grid_search_{}.log".format(i)
            sys.stdout = Logger(self.log_fname)


            print "Using params:"
            gparam.print_params()

            
            cv = CrossValidator(
                                  gparam,
                                  self.nuc_data,
                                  self.cv_save_dirs[i],
                                  seed = self.seed,
                                  k_folds = self.k_folds,
                                  test_frac = self.test_frac
                               )

            cv.run()

            gparam_save_dir = os.path.basename(self.cv_save_dirs[i])
            cv.plot_auroc_auprc(self.fig_title_prefix,
                                fname_prefix = gparam_save_dir,
                                include_fill=False)


            _, cv_avg_test_results = cv.calc_avg_k_metrics()
            self.cv_test_results.append(cv_avg_test_results)
        self.best_model_params, self.best_model_name = self._best_test_params("auroc")
        self.save_data()

    



    def save_data(self):
        """
        Save the following (in order!) so that self.run() does not
        need to be recalled in order to load important information:
        self.cv_test_results-- A list of the average test metrics for each cross validation run
        self.best_model
        """
        with open(self.grid_data_file,'w') as of:
            pickle.dump(self.cv_test_results,of)
            best_json = self.best_model_dir+os.sep+self.best_model_name+'.json'
            self.best_model_params.save_as_json(best_json)
            
    
    def load_data(self):
        with open(self.grid_data_file,'r') as lf:
            self.cv_test_results = pickle.load(lf)
            self.best_model_params, self.best_model_name = self._best_test_params("auroc")
                  
                
    def _best_test_params(self,metric="auroc"):
        """
        Must be called after self.load_data() or self.run() which populates
        cv_test_results, and best_model_params_folder
        Get the ModelParams corresponding to the best
        performing hyperparameter set evaluated by 3 fold cross_validation

        :param metric: string key for a given metric (such as "auroc" or "f1_score")
        :returns: ModelParams object
        :rtype: ModelParams
        """
        
        if self.cv_test_results == []:
            print "No test results. Please run self.run() or self.load_data() first"
            return None

        best_val = 0
        #Find the highest scoring model
        # and return corresponding ModelParams
        num_results=  len(self.cv_test_results)
        for i in range(num_results):
            best_val = max( self.cv_test_results[i][metric],best_val)
        for i in range(num_results):
            if self.cv_test_results[i][metric] == best_val:
                self.grid_params_list[i].print_params()
                best_model_name = 'best_model_gparam_cv{}'.format(i)
                print "Best {} model in gparam_cv{}".format(metric,i)
                self.best_model_json = self.best_model_dir+os.sep+best_model_name+\
                                                                      '.json'
                #self.grid_params_list[i].save_as_json(self.best_model_json)
                return self.grid_params_list[i],best_model_name



                    
    def best_binary_classifier(self,sess,metric="auroc"):
        """
        This must be run after self.run() or self.load()
        
        Get the best model for a given metric from all parameter sets.
        Then train that model using the entire training set
    
        :metric: string key for a given metric (such as "auroc" or "f1_score")
        :returns: A NucBinaryClassifier object
        :rtype: NucBinaryClassifier
        """

        bparams = self.best_model_params
        
        all_indices = range(self.nuc_data.num_records)
        
        train_batcher = DataBatcher(self.nuc_data,all_indices)

        test_batcher = None

        self.best_classifier = NucBinaryClassifier(sess,
                                         train_batcher,
                                         test_batcher, 
                                         bparams.num_epochs,
                                         bparams.learning_rate,
                                         bparams.batch_size,
                                         bparams.seq_len,
                                         self.best_model_dir, 
                                         bparams.keep_prob,
                                         bparams.beta1,
                                         bparams.inference_method_key)
        self.best_classifier.build_model()
        self.best_classifier.train()
        #self.best_classifier.plot_test_epoch_vs_metric('auroc',
        #                                'best_model',
        #                            save_plot=True)
        
        return (self.best_classifier,bparams)


                
class BayesianOptimSearch(object):
    def __init__(self):
        pass
    
