from modelparams import *
from crossvalidator import CrossValidator
import sys
import os
import uuid
from logger import Logger
from databatcher import DataBatcher 


class GridSearch(object):
    def __init__(self,grid_params_obj,nuc_data,gs_save_dir,
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
        self.cv_results = []

    def run(self):
        self.log_fname = self.save_dir+os.sep+"grid_search_{}.log".format(str(uuid.uuid4))
        sys.stdout = Logger(self.log_fname)

        for gparam in self.grid_params_list:
            cv = CrossValidator(
                                  gparam,
                                  self.nuc_data,
                                  self.save_dir,
                                  seed = self.seed,
                                  nn_method = gparam.inference_method,
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
            best_val = max( cv_test_results[i][metric],best_val)
        for i in range(num_results):
            if cv_test_results[i][metric] == best_val:
                return self.grid_params_list[i]

                    
    def best_binary_classifier(self,metric="auroc"):
        """Get the best NucBinaryClassifier model for a given metric
    
        :metric: string key for a given metric (such as "auroc" or "f1_score")
        :returns: A NucBinaryClassifier object
        :rtype: NucBinaryClassifier
        """
        bparams = self.best_model_params(metric)


        perm_indices = np.random.RandomState(self.seed).\
                    permutation(range(self.nuc_data.num_records))
        test_frac = .2
        test_size = int(self.nuc_data.num_records*test_frac)
        train_size = self.nuc_data.num_records - test_size
        test_indices = perm_indices[0:int(self.nuc_data.num_records*test_frac)]
        train_indices = np.setdiff1d(perm_indices,test_indices)
        
        train_batcher = DataBatcher(self.nuc_data,train_indices)
        test_batcher = DataBatcher(self.nuc_data,test_indices)

        fsave_dir = self.save_dir+os.sep+self.save_dir
        sess = tf.Session()
        classifier = NucBinaryClassifier(sess,
                                         train_batcher,
                                         test_batcher, 
                                         bparams.num_epochs,
                                         bparams.learning_rate,
                                         bparams.batch_size,
                                         bparams.seq_len,
                                         fsave_dir, 
                                         bparams.keep_prob,
                                         bparams.beta1)
        classifier.build_model(bparams.inference_model)
        classifier.train()

        return (classifier,best_model_params,sess)


                
class BayesianOptimSearch(object):
    def __init__(self):
        pass
    
