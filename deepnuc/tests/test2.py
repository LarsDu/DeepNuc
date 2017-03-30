import sys
import os.path
sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import tensorflow as tf
import numpy as np

from nucdata import *
from nucclassifier import NucClassifier
from databatcher import DataBatcher 
from modelparams import *
from logger import Logger
import nucconvmodel
from crossvalidator import CrossValidator


def main():
    params = ModelParams( training_file=None,
                          testing_file=None,
                          num_epochs=20,
                          learning_rate=1e-4,
                          batch_size=24,
                          seq_len=600,
                          keep_prob=0.5,
                          beta1=0.9 )
    cv_k_folds = 3
    cv_test_frac = 0.15
        
    test_classifier(params)
    #test_cross_validation()
    test_relevance()
    #test_mutation_map()




def test_classifier(params):
    sys.stdout = Logger('logt2.log')
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

    save_dir = "example_data/test2"

    
    with tf.Session() as sess:
        nc_test = NucClassifier(sess,
                                train_batcher,
                                test_batcher,
                                params.num_epochs,
                                params.learning_rate,
                                params.batch_size,
                                params.seq_len,
                                save_dir,
                                params.keep_prob,
                                params.beta1)

        nc_test.build_model(nucconvmodel.inferenceA)
        nc_test.train()


def test_relevance():
    pass
        
def test_cross_validation():
    pass

def test_mutation_map():
    pass

if __name__ == "__main__":
    main()
