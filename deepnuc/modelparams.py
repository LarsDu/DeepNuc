import json
import numpy as np



class ModelParams(object):
    
    def __init__(self,
                 training_file,
                 testing_file,
                 num_epochs,
                 learning_rate,
                 batch_size,
                 seq_len,
                 k_folds,
                 test_frac,
                 keep_prob = 0.5,
                 beta1 = 0.9):


        ##Training parameters
        if training_file == '' or training_file == 'None':
            self.training_file = None
        else:
            self.training_file = training_file

        if testing_file == '' or testing_file == 'None':
            self.testing_file = None
        else:
            self.testing_file = testing_file

        self.num_epochs = int(num_epochs)
        self.keep_prob = float(keep_prob)
        self.learning_rate = np.float32(learning_rate)
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.k_folds = int(k_folds)
        self.test_frac = float(test_frac)
        self.beta1 = float(beta1)
        


class JsonModelParams(Params):
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
            training_file = self.json_path+os.sep+data['files']['training_file']
            testing_file = self.json_path+os.sep+data['files']['testing_file']

            if training_file == '' or training_file == 'None':
                self.training_file = None
            else:
                self.training_file = training_file

            if testing_file == '' or testing_file == 'None':
                self.testing_file = None
            else:
                self.testing_file = testing_file
            #Options
            self.save_dir=self.json_path+os.sep+os.path.basename(data['options']['save_dir'])
            
            
            ##Training parameters
            num_epochs = int(data['training_params']['num_epochs'])
            keep_prob = float(data['training_params']['keep_prob'])
            #self.num_iterations = int(data['training_params']['num_iterations'])
            learning_rate = np.float32(data['training_params']['learning_rate'])
            seq_len = int(data['training_params']['seq_len'])
            batch_size = int(data['training_params']['batch_size'])
            k_folds = data['training_params']['k_folds']
            test_frac = data['training_params']['test_frac']

            #self.custom_model_params = data['training_params']['custom_model_params']
            #if (self.custom_model_params == "" or
            #    self.custom_model_params == None):
            #    self.custom_model_params = []
            
            #Make sure k_folds values make sense
            if (self.k_folds>0) and (1./self.k_folds < self.k_validation_test_frac):
                print "ERROR!!"
                print('test_frac ',self.k_validation_test_frac,' too large for k=',
                params.k_folds,' fold.') 

                
            ###Create custom args dict for custom parameters
            #Convert comma delimited custom_model_params to list
            #print "testparams", self.custom_model_params==""
            if self.custom_model_params != []:
                self.custom_model_params = self.custom_model_params.strip().split(',')

                
            self.custom_args_dict = {}
            for str_arg in self.custom_model_params:
                key,value = str_arg.split('=')
                self.custom_args_dict[str(key)]=int(value)

            print "Custom model params: ",self.custom_args_dict

