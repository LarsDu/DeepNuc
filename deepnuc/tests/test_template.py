import sys
import os.path
sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)))

import tensorflow as tf
import numpy as np

from nucdata import *
from nucbinaryclassifier import NucBinaryClassifier
from databatcher import DataBatcher 
from modelparams import *
from logger import Logger
import nucconvmodel
from crossvalidator import CrossValidator

def main():
    params = ModelParams(
                          seq_len=600,
                          num_epochs=1000,
                          learning_rate=1e-4,
                          batch_size=24,
                          keep_prob=0.5,
                          beta1=0.9,
                          concat_revcom_input=False,
                          inference_method_key="inferenceA"
                        )

    save_dir= 'sim_cv_test1'
    nuc_data = NucDataFastaMem(['600_pos1_dinuc.fa','600_pos1_3000ex.fa'],params.seq_len)


    #tbatcher = DataBatcher(nuc_data,range(0,1000))


    #test_cross_validation(nuc_data,params,save_dir,k_folds=3,test_frac=0.2)
    #test_classifier(nuc_data,params,save_dir,'train')
    test_classifier(nuc_data,params,save_dir,'visualize')



def test_classifier(nuc_data,params,save_dir,mode='mode',test_frac = .2,seed=547125):
    sys.stdout = Logger(save_dir+os.sep+mode+".log")
    print "Test classifier"

    perm_indices = np.random.RandomState(seed).\
                    permutation(range(nuc_data.num_records))
   
    test_size = int(nuc_data.num_records*test_frac)
    train_size = nuc_data.num_records - test_size
    test_indices = perm_indices[0:int(nuc_data.num_records*test_frac)]
    train_indices = np.setdiff1d(perm_indices,test_indices)
    train_batcher = DataBatcher(nuc_data,train_indices)
    test_batcher = DataBatcher(nuc_data,test_indices)



    if mode =="train":
        keep_prob = params.keep_prob
    elif mode =="visualize":
        keep_prob=1.0
         
    with tf.Session() as sess:
        nc_test = NucBinaryClassifier(sess,
                                train_batcher,
                                test_batcher,
                                params.num_epochs,
                                params.learning_rate,
                                params.batch_size,
                                params.seq_len,
                                save_dir,
                                keep_prob,
                                params.beta1,
                                params.concat_revcom_input,
                                nn_method_key=params.inference_method_key
                                )

        nc_test.build_model()

        if mode == 'train':
            nc_test.train()
        elif mode == 'visualize':
            all_batcher = DataBatcher(nuc_data,range(nuc_data.num_records))
            nc_test.plot_relevance_heatmap_from_batcher(test_batcher,420)
            nc_test.plot_alipanahi_mutmap_from_batcher(test_batcher,420)
            nc_test.plot_relevance_heatmap_from_batcher(test_batcher,423)
            nc_test.plot_alipanahi_mutmap_from_batcher(test_batcher,423)
            nc_test.plot_relevance_heatmap_from_batcher(test_batcher,123)
            nc_test.plot_alipanahi_mutmap_from_batcher(test_batcher,123)
            nc_test.plot_relevance_heatmap_from_batcher(test_batcher,39)
            nc_test.plot_alipanahi_mutmap_from_batcher(test_batcher,39)

        
       
def test_cross_validation(nuc_data,params,save_dir,k_folds=3,test_frac=0.2,seed=1912858):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print "Test cross-validation"
    logf = save_dir+os.sep+'cv.log'
    sys.stdout = Logger(logf)
    
    
    cv = CrossValidator( 
                        params, 
                        nuc_data,
                        save_dir,
                        seed=seed,
                        k_folds=3,
                        test_frac=0.15)
    cv.run()
    
    cv.calc_avg_k_metrics()
   
    

if __name__ == "__main__":
    main()
    
 
