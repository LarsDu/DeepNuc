from modelparams import *
from crossvalidator import CrossValidator
import sys
import os
from logger import Logger
from databatcher import DataBatcher 
from nucbinaryclassifier import NucBinaryClassifier
from nucregressor import NucRegressor

class GridSearch(object):
    def __init__(self,
                 grid_params_obj,
                 nuc_data,
                 gs_save_dir,
                 seed = 1212516,
                 k_folds = 3,
                 test_frac =0.20):
        
        self.nuc_data = nuc_data
        self.grid_params = grid_params_obj
        self.grid_params_list = grid_params_obj.grid_params_list
        self.save_dir = gs_save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        
        self.seed = seed
        self.k_folds =k_folds
        self.test_frac = test_frac
        self.cv_test_results = []

    def run(self):

        for i,gparam in enumerate(self.grid_params_list):
            gparam_save_dir = 'gparam_cv{}'.format(i)
            cv_save_dir = self.save_dir+os.sep+gparam_save_dir
            self.log_fname = cv_save_dir+os.sep+"grid_search_{}.log".format(i)
            sys.stdout = Logger(self.log_fname)

            print "Using params:"
            gparam.print_params()

            
            cv = CrossValidator(
                                  gparam,
                                  self.nuc_data,
                                  cv_save_dir,
                                  seed = self.seed,
                                  k_folds = self.k_folds,
                                  test_frac = self.test_frac
                               )

            cv.run()
            _, cv_avg_test_results = cv.calc_avg_k_metrics()
            self.cv_test_results.append(cv_avg_test_results)


    def best_model_params(self,metric="auroc"):
        """Get the ModelParams corresponding to the best
        performing hyperparameter set evaluated by 3 fold cross_validation

        :param metric: string key for a given metric (such as "auroc" or "f1_score")
        :returns: ModelParams object
        :rtype: ModelParams

        """
        
        if self.cv_test_results == []:
            print "No test results. Please run GridSearch.run() first"
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
                return self.grid_params_list[i]

                    
    def best_binary_classifier(self,sess,metric="auroc"):
        """
        Get the best model for a given metric from all parameter sets.
        Then train that model using the entire training set
    
        :metric: string key for a given metric (such as "auroc" or "f1_score")
        :returns: A NucBinaryClassifier object
        :rtype: NucBinaryClassifier
        """

        
        bparams = self.best_model_params(metric)

        perm_indices = np.random.RandomState(self.seed).\
                    permutation(range(self.nuc_data.num_records))
        #test_frac = .2
        #test_size = int(self.nuc_data.num_records*test_frac)
        #train_size = self.nuc_data.num_records - test_size
        #test_indices = perm_indices[0:int(self.nuc_data.num_records*test_frac)]
        #train_indices = np.setdiff1d(perm_indices,test_indices)
        
        train_batcher = DataBatcher(self.nuc_data,perm_indices)
        #test_batcher = DataBatcher(self.nuc_data,test_indices)
        test_batcher = None
        self.best_classifier_folder = 'best_model'
        self.best_classifier_dir = self.save_dir+os.sep+self.best_classifier_folder

        classifier = NucBinaryClassifier(sess,
                                         train_batcher,
                                         test_batcher, 
                                         bparams.num_epochs,
                                         bparams.learning_rate,
                                         bparams.batch_size,
                                         bparams.seq_len,
                                         self.best_classifier_dir, 
                                         bparams.keep_prob,
                                         bparams.beta1,
                                         bparams.inference_method_key)
        classifier.build_model()
        classifier.train()
        return (classifier,bparams)


                
class BayesianOptimSearch(object):
    def __init__(self):
        pass
    
