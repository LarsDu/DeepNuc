import json
import DreamDataInput
import os

def main_test():
    json_file = "EGR1_train.json"
    
    data_params = JsonData(json_file)
    print data_params.train_hdf_file
    print data_params.genome_file
    print data_params.test_cell_types
    print type(data_params.seq_len)
    print type(data_params.learning_rate)
    print data_params
    data_input = DreamDataInput.DreamDataLoaderJson(json_file)

class JsonParams:
    def __init__(self,json_file):
        self.json_filename = json_file
        with open(self.json_filename,'r') as jf:
            data= json.load(jf)
            self.train_hdf5_file = data['files']['train_hdf5_file']
            #self.eval_hdf5_file = data['files']['eval_hdf5_file']
            #self.test_hdf5_file = data['files']['test_hdf5_file']
            self.train_coords_file = data['files']['train_coords_file']
            self.test_coords_file= data['files']['test_coords_file']
            self.eval_coords_file = data['files']['eval_coords_file']

            #l=[self.train_hdf5_file,self.eval_hdf5_file,self.train_coords_file,
            #   self.test_coords_file,self.eval_coords_file]:
            l = [ self.train_hdf5_file,self.train_coords_file,self.test_coords_file,
                  self.eval_coords_file]
            for f in l: 
                if f != None:
                    f =os.path.abspath(f)
                    

            
            
            self.genome_file = os.path.abspath(data['files']['genome_file'])
            self.genome_annotation_file = os.path.abspath(data['files']['genome_annotation_file'])
            self.genome_chromosome_sizes_file = os.path.abspath(
                                                  data['files']['genome_chromosome_sizes_file'])
            self.dna_shape_dir = data['files']['dna_shape_dir']
            self.dnase_seq_dir = data['files']['dnase_seq_dir']
            self.dnase_relaxed_peaks_dir = data['files']['dnase_relaxed_peaks_dir']
            self.rna_seq_dir = data['files']['rna_seq_dir']
            self.chip_seq_dir = data['files']['chip_seq_dir']
            #Strip dirs of trailing slashes
            dirs = [self.dna_shape_dir,self.dnase_seq_dir,self.dnase_relaxed_peaks_dir,
                 self.rna_seq_dir,self.chip_seq_dir]
            for fdir in dirs:
                if fdir != None:
                    fdir = os.path.abspath(fdir.rstrip('/').rstrip('\\'))
                    
            
            self.chip_seq_labels_file = os.path.abspath(data['files']['chip_seq_labels_file'])
            #self.mode = data['training_params']['mode']
            self.tfactor_name = data['training_params']['tfactor_name']
            self.train_cell_types = data['training_params']['train_cell_types']
            self.test_cell_types = data['training_params']['test_cell_types']
            self.eval_cell_types = data['training_params']['eval_cell_types']
            if self.train_cell_types != None:
                self.train_cell_types = self.train_cell_types.strip().split(',')
            if self.test_cell_types != None:
                self.test_cell_types = self.test_cell_types.strip().split(',')
            if self.eval_cell_types != None:
                self.eval_cell_types = self.eval_cell_types.strip().split(',')
            
                
            self.eval_predictions_file = 'F.'+self.tfactor_name+'.'+'.'.join(self.eval_cell_types)+'.tab'
            self.max_train_steps = data['training_params']['max_train_steps']
            self.learning_rate = data['training_params']['learning_rate']
            self.batch_size = int(data['training_params']['batch_size'])
            self.seq_len = int(data['training_params']['seq_len'])
            self.chrom_window = int(data['training_params']['chrom_window'])
            self.chrom_pool_size = int(data['training_params']['chrom_pool_size'])
            self.pooled_chrom_window = int(self.chrom_window//self.chrom_pool_size)
            if self.chrom_window% self.chrom_pool_size !=0:
                print "Warning! chrom_window must be evenly divisible by chrom_pool_size!"            
            #self.checkpoint_dir = data['training_params']['checkpoint_dir']
            self.custom_model_params = data['training_params']['custom_model_params']
            if self.custom_model_params != None:
                self.custom_model_params = self.custom_model_params.strip().split(',')
            self.k_folds = data['training_params']['k_folds']
            self.k_validation_test_frac = data['training_params']['k_validation_test_frac']
            self.save_dir=data['save']['save_dir']
           
           




if __name__=="__main__":
    main_test()
