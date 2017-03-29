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
    #tf1()
    #tf2()
    #tf3()
    #tf4()
    #tf6()
    tf7()
    
def tf1():
    data_list = ["example_data/WS230_tss_list.fa"]
    #data_list = ["example_data/auto_dinuc.fa","example_data/WS230_tss_list.fa"]
    nuc_data = NucFastaMemory(data_list,seq_len=600)
    print nuc_data.num_records
    print nuc_data.pull_index_nucs(245)
    print nuc_data.pull_index_onehot(245)

def tf2():
    print "\n\ntf2\n"
    fname = "example_data/worm_tss_nib.h5"
    nuc_data = NucHdf5(fname)
    print nuc_data.num_records
    print nuc_data.pull_index_nucs(245+nuc_data.num_records//2)
    print nuc_data.pull_index_onehot(245+nuc_data.num_records//2)

    
def tf3():
    sys.stdout = Logger('log1.log')
    print "\n\ntf3\n"
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

    print nuc_data.pull_index_nucs(45)
    print nuc_data.pull_index_nucs(46)
    print nuc_data.pull_index_nucs(47)
    print nuc_data.pull_index_nucs(5000)

    print train_batcher.pull_batch(33)
    

def tf4():
    sys.stdout = Logger('log1.log')
    print "\n\ntf3\n"
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


    params = ModelParams(training_file=None,
                    testing_file=None,
                    num_epochs=10,
                    keep_prob=0.5,
                    learning_rate=1e-4,
                    seq_len=600,
                    batch_size=24,
                    k_folds =0,
                    test_frac=0.2)

    save_dir = "example_data/example_train/checkpoints"


    with tf.Session() as sess:
        nc_test = NucClassifier( sess,
                                 train_batcher,
                                 test_batcher,
                                 params,
                                 save_dir)


        nc_test.build_model(nucconvmodel.inferenceA)
        nc_test.train()


def tf5():
    sys.stdout = Logger('example_data/log2.log')
    print "\n\ntf4\n"
    fname = "example_data/worm_tss_nib.h5"
    nuc_data = NucHdf5(fname)

    seed = 64
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


    params = Params(training_file=None,
                    testing_file=None,
                    num_epochs=100,
                    keep_prob=0.5,
                    learning_rate=1e-4,
                    seq_len=600,
                    batch_size=24,
                    k_folds =0,
                    test_frac=0.2)

    save_dir = "example_data/example_train/"


    with tf.Session() as sess:
        nc_test = NucClassifier( sess,
                                 train_batcher,
                                 test_batcher,
                                 params,
                                 save_dir)


        nc_test.build_model(nucconvmodel.inferenceB)
        nc_test.train()


        
def tf6():
    sys.stdout = Logger('example_data/cv_test.log')
    print '\n\ntf6\n'
    fname = "example_data/worm_tss_nib.h5"
    nuc_data = NucHdf5(fname)

    params = Params(training_file = None,
                    testing_file = None,
                    num_epochs = 5,
                    keep_prob = 0.5,
                    learning_rate=1e-4,
                    seq_len = 600,
                    batch_size = 24,
                    k_folds =3,
                    test_frac=0.15)
    cv_save_dir = "example_data/cv_test"

    with tf.Session() as sess:

        
        cv = CrossValidator( sess,
                             params,
                             nuc_data,
                             cv_save_dir,
                             nn_method=nucconvmodel.inferenceA)

def tf7():
    sys.stdout = Logger('example_data/log_tf7.log')
    print "\n\ntf7\n"
    fname = "example_data/worm_tss_nib.h5"
    nuc_data = NucHdf5(fname)

    seed = 64
    perm_indices = np.random.RandomState(seed).\
                    permutation(range(nuc_data.num_records))
    test_frac = .2
    test_size = int(nuc_data.num_records*test_frac)
    train_size = nuc_data.num_records - test_size
    test_indices = perm_indices[0:int(nuc_data.num_records*test_frac)]
    train_indices = np.setdiff1d(perm_indices,test_indices)

    print test_indices.shape
    print train_indices.shape


    all_batcher = DataBatcher(nuc_data,range(nuc_data.num_records))
    train_batcher = DataBatcher(nuc_data,train_indices)
    test_batcher = DataBatcher(nuc_data,test_indices)


    params = Params(training_file=None,
                    testing_file=None,
                    num_epochs=10,
                    keep_prob=0.5,
                    learning_rate=1e-4,
                    seq_len=600,
                    batch_size=24,
                    k_folds =0,
                    test_frac=0.2)

    save_dir = "example_data/example_train"

    
    mode = 'relevance'
    with tf.Session() as sess:
        nc_test = NucClassifier( sess,
                                 train_batcher,
                                 test_batcher,
                                 params,
                                 save_dir)


        nc_test.build_model(nucconvmodel.inferenceB)
        if mode == 'train':
            nc_test.train()
        elif mode == 'relevance':
            nc_test.relevance_batch_by_index(all_batcher,100,1)

        
        









        
    
if __name__ == "__main__":
    main()
