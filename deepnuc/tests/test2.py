import sys
import os.path
sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

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
                          num_epochs=35,
                          learning_rate=1e-4,
                          batch_size=24,
                          keep_prob=0.5,
                          beta1=0.9,
                          concat_revcom_input=False,
                          inference_method_key="inferenceA"
                        )
        
    #test_classifier(params,"train")
    #test_classifier(params,"relevance")

    test_cross_validation(params)
    
    #test_mutation_map()




def test_classifier(params,mode="train"):

    save_dir = "example_data/may_lab_meeting"
    sys.stdout = Logger(save_dir+os.sep+mode+".log")
    print "Test classifier"

    fname = "example_data/worm_tss_nib.h5"
    nuc_data = NucHdf5(fname)

    seed = 12415
    perm_indices = np.random.RandomState(seed).\
                    permutation(range(nuc_data.num_records))
    test_frac = .2
    test_size = int(nuc_data.num_records*test_frac)
    train_size = nuc_data.num_records - test_size
    test_indices = perm_indices[0:int(nuc_data.num_records*test_frac)]
    train_indices = np.setdiff1d(perm_indices,test_indices)

    print test_indices.shape
    print train_indices.shape


    train_batcher = DataBatcher(nuc_data,train_indices)
    test_batcher = DataBatcher(nuc_data,test_indices)

    if mode =="train":
        keep_prob = params.keep_prob
    elif mode =="relevance":
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
                                params.beta1)

        #nc_test.build_model(params.inference_model)

        if mode == 'train':
            nc_test.train()
        elif mode == 'relevance':
            all_batcher = DataBatcher(nuc_data,range(nuc_data.num_records))
            #nc_test.relevance_batch_by_index(all_batcher,5004)
            #nc_test.relevance_batch_by_index(all_batcher,5007)
            #nc_test.relevance_batch_by_index(all_batcher,5009)
            nc_test.relevance_batch_by_index(all_batcher,8000)
            #nc_test.eval_train_test()



    
def test_cross_validation(params):
    cv_save_dir = "example_data/cv_test2"
    if not os.path.exists(cv_save_dir):
        os.makedirs(cv_save_dir)
        
    print "Test cross-validation"
    logf = cv_save_dir+os.sep+"cv_test2.log"
    sys.stdout = Logger(logf)


    fname = "example_data/worm_tss_nib.h5"
    nuc_data = NucHdf5(fname)

    seed = 12415

    test_frac = .2
    

    cv = CrossValidator( 
                        params, 
                        nuc_data,
                        cv_save_dir,
                        seed=seed,
                        k_folds=3,
                        test_frac=0.15)
    cv.run()
    
    cv.calc_avg_k_metrics()
    
def test_mutation_map():
    pass

if __name__ == "__main__":
    main()
