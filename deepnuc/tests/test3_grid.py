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
from hypersearch import GridSearch

def main():
    gparams = GridParams(
                          seq_len= 600,
                          num_epochs=[25,50],
                          learning_rate=[1e-4,1e-5],
                          batch_size=[24],
                          keep_prob=[0.5],
                          beta1=[0.9],
                          concat_revcom_input = [True,False])

    print "Grid parameter values:"
    print [p.print_param_values() for p in gparams.grid_params_list]
    print "\n"
    
    fname = "example_data/worm_tss_nib.h5"
    nuc_data = NucHdf5(fname)

    gs_save_dir = "example_data/grid_search_test1"
    gsearch = GridSearch(
                         gparams,
                         nuc_data,
                         gs_save_dir,
                         seed = 125561,
                         k_folds=3,
                         test_frac=.2
                         )
    
    gsearch.run()
    best_classifier,best_model_params,sess = gsearch.best_binary_classifier("auroc")
    sess.close()

    
if __name__ == "__main__":
    main()

    
