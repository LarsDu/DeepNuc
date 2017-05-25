import json
import numpy as np
from collections import OrderedDict
import itertools
import nucconvmodel
import json
import os


class GridParams(object):

    def __init__(self,
                 seq_len,
                 num_epochs=[50],
                 learning_rate=[1e-4],
                 batch_size=[24],
                 keep_prob = [0.5],
                 beta1 = [0.9],
                 concat_revcom_input=[False],
                 inference_method_key=["inferenceA"]):

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
                                      ('inference_method_key',set(inference_method_key)),
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
                 inference_method_key="inferenceA",
                 json_file=None):


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
        self.inference_method_key = inference_method_key
        self.inference_method = nucconvmodel.methods_dict[inference_method_key]

        
        #self.k_folds = int(k_folds)
        #self.test_frac = float(test_frac)
        self.populate_param_dict()
        self.json_file = json_file
        
        
    @classmethod
    def init_json(cls,json_file):
        print "Parsing json file",json_file
        with open (json_file,'r') as jf:
            data = json.load(jf)
            num_epochs = int(data['num_epochs'])
            keep_prob = float(data['keep_prob'])
            #num_iterations = int(data['num_iterations'])
            learning_rate = np.float32(data['learning_rate'])
            seq_len = int(data['seq_len'])
            batch_size = int(data['batch_size'])
            beta1 = data['beta1']
            concat_revcom_input = data['concat_revcom_input']
            inference_method_key = data['inference_method_key']

            
        return cls( seq_len,
                    num_epochs,
                    learning_rate,
                    batch_size,
                    keep_prob,
                    beta1,
                    concat_revcom_input,
                    inference_method_key,
                    json_file
                    )
        
    
    def extract_json(self,json_file):
        """Avoid using this in favor of ModelParams.init_json(json_file)"""
        self.json_file = os.path.abspath(json_file)
        #self.json_path = os.path.dirname(os.path.abspath(self.json_filename))
        print "Parsing json file",self.json_filename
        with open (self.json_filename,'r') as jf:
            data = json.load(jf)
            self.num_epochs = int(data['num_epochs'])
            self.keep_prob = float(data['keep_prob'])
            #self.num_iterations = int(data['num_iterations'])
            self.learning_rate = np.float32(data['learning_rate'])
            self.seq_len = int(data['seq_len'])
            self.batch_size = int(data['batch_size'])
            #self.k_folds = data['k_folds']
            #self.test_frac = data['test_frac']
            self.beta1 = data['beta1']
            self.concat_revcom_input = data['concat_revcom_input']
            self.inference_method_key = data['inference_method_key']
            self.inference_method = nucconvmodel.methods_dict[self.inference_method_key]
        self.populate_param_dict()
        
    def populate_param_dict(self):
        self.params_dict = OrderedDict([
                                      ('seq_len',int(self.seq_len)),
                                      ('num_epochs',int(self.num_epochs)),
                                      ('learning_rate',float(self.learning_rate)),
                                      ('batch_size',int(self.batch_size)),
                                      ('keep_prob',float(self.keep_prob)),
                                      ('beta1',float(self.beta1)),
                                      ('concat_revcom_input',self.concat_revcom_input),
                                      ('inference_method_key',self.inference_method.__name__),
                                      ])
       
        
    def print_param_values(self):
        print self.params_dict.values()
        
    def print_params(self):
        for k,v in self.params_dict.viewitems():
            print "{}:\t{}".format(k,v)

        print "\n"
        
    def save_as_json(self,out_file):
        print "Saving ModelParams in", out_file
        with open(out_file,'w') as of:
            #print self.params_dict["inference_method_key"]
            json.dump(self.params_dict,of)

    
        
