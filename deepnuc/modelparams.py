import json
import numpy as np
from collections import OrderedDict
import itertools
import nucconvmodel

class GridParams(object):

    def __init__(self,
                 seq_len,
                 num_epochs=[50],
                 learning_rate=[1e-4],
                 batch_size=[24],
                 keep_prob = [0.5],
                 beta1 = [0.9],
                 concat_revcom_input=[False],
                 inference_method=[nucconvmodel.inferenceA]):

        '''
        Pass parameters as a with values as list
        IE: num_epochs=[50,40], learning_rate=[1e-4,1e-5]        
        '''
        self.param_dict = OrderedDict([
                                      ('num_epochs',set(num_epochs)),
                                      ('learning_rate',set(learning_rate)),
                                      ('batch_size',set(batch_size)),
                                      ('keep_prob',set(keep_prob)),
                                      ('beta1',set(beta1)),
                                      ('concat_revcom_input',set(concat_revcom_input)),
                                      ('inference_method',set(inference_method)),
                                     ])
        
        cartesian_prod = itertools.product(*self.param_dict.viewvalues())
        

        self.grid_params_list = []
        for param_tuple in cartesian_prod:
            self.grid_params_list.append(ModelParams(seq_len,*param_tuple))
        self.num_perms = len(self.grid_params_list)

        
        
        
class ModelParams(object):
    
    def __init__(self,
                 seq_len,
                 num_epochs,
                 learning_rate,
                 batch_size,
                 keep_prob = 0.5,
                 beta1 = 0.9,
                 concat_revcom_input=False,
                 inference_method=nucconvmodel.inferenceA):


        ##Training parameters
        '''
        if training_file == '' or training_file == 'None':
            self.training_file = None
        else:
            self.training_file = training_file

        if testing_file == '' or testing_file == 'None':
            self.testing_file = None
        else:
            self.testing_file = testing_file
        '''
        self.num_epochs = int(num_epochs)
        self.learning_rate = np.float32(learning_rate)
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)

        self.keep_prob = float(keep_prob)
        self.beta1 = float(beta1)
        self.concat_revcom_input = concat_revcom_input
        self.inference_method = inference_method
        #self.k_folds = int(k_folds)
        #self.test_frac = float(test_frac)
        self.params_dict = OrderedDict([
                                      ('num_epochs',self.num_epochs),
                                      ('learning_rate',self.learning_rate),
                                      ('batch_size',self.batch_size),
                                      ('keep_prob',self.keep_prob),
                                      ('beta1',self.beta1),
                                      ('concat_revcom_input',self.concat_revcom_input),
                                      ('inference_method',self.inference_method),
                                      ])
        

    def print_param_values(self):
        print self.params_dict.values()
        
    def print_params(self):
        for k,v in self.params_dict.viewitems():
            print "{}:\t{}".format(k,v)

        print "\n"
        
        

class JsonModelParams(ModelParams):
    """
    A class for processing training and model variables defined
    in a Json file. User can define this file for setting up training
    runs
    """
    
    def __init__(self,json_file):
        self.json_filename = os.path.abspath(json_file)
        self.json_path = os.path.dirname(os.path.abspath(self.json_filename))
        print "Parsing json file",self.json_filename
        with open (self.json_filename,'r') as jf:
            data = json.load(jf)
            ##'mode' should be set by commandline gflags
                       
            #Files
            #Location of files relative to json file
            #training_file = self.json_path+os.sep+data['files']['training_file']
            #testing_file = self.json_path+os.sep+data['files']['testing_file']
            
            #save_dir=self.json_path+os.sep+os.path.basename(data['files']['save_dir'])
            ##Training parameters
            num_epochs = int(data['training_params']['num_epochs'])
            keep_prob = float(data['training_params']['keep_prob'])
            #self.num_iterations = int(data['training_params']['num_iterations'])
            learning_rate = np.float32(data['training_params']['learning_rate'])
            seq_len = int(data['training_params']['seq_len'])
            batch_size = int(data['training_params']['batch_size'])
            k_folds = data['training_params']['k_folds']
            test_frac = data['training_params']['test_frac']
            beta1 = data['training_params']['beta1']
            concat_revcom_input = data['training_params']['concat_revcom_input']
        #Initialize parent class
        super(JsonModelParams,
              self,
              seq_len,
              num_epochs,
              learning_rate,
              batch_size,
              keep_prob,
              beta1,
              concat_revcom_input).__init__()
                
          

