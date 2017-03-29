import numpy as np
import json
import DuBioTools as dbt
import os
import copy

import gzip
import pysam
from Bio import SeqIO
import h5py
import DinucShuffle

"""
This file contains the following classes:

    JsonParams(json_file): A parser for a file specifying training

    JsonDataCollection(json_param_obj): A faster way of initializing a DataCollection
    

    FastaReader([fasta_file],seq_len)

    ConvertToHdf5(reader_list,labels)
    
    InputCollection(json_params): Takes hdf5 files. Permutes batch draws, and
    tracks epochs. Reshuffles data at end of each epoch

    EpochTracker(num_examples): Keeps track of training epochs. Sends signal for
                                reshuffling data     

    
"""



class JsonParams:
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
            eval_file = self.json_path+os.sep+data['files']['eval_file']

            if training_file == '' or training_file == 'None':
                self.training_file = None
            else:
                self.training_file = training_file

            if eval_file == '' or eval_file == 'None':
                self.eval_file = None
            else:
                self.eval_file = eval_file
            #Options
            self.save_dir=self.json_path+os.sep+os.path.basename(data['options']['save_dir'])
            
            
            ##Training parameters
            self.num_iterations = int(data['training_params']['num_iterations'])
            self.learning_rate = np.float32(data['training_params']['learning_rate'])
            self.seq_len = int(data['training_params']['seq_len'])
            self.batch_size = int(data['training_params']['batch_size'])
            self.custom_model_params = data['training_params']['custom_model_params']
            if (self.custom_model_params == "" or
                self.custom_model_params == None):
                self.custom_model_params = []
            self.k_folds = data['training_params']['k_folds']
            self.k_validation_test_frac = data['training_params']['k_validation_test_frac']

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





class ConvertToHdf5:

    def __init__(self,reader_list,output_fname = None,labels=None):

        self.reader_list = reader_list
        self.output_fname = output_fname
        self.output_files =  []
        self.seq_len = reader_list[0].seq_len
        #self.save_genidx = save_genidx
        
        if labels == None:
            self.labels = dbt.numeric_labels_to_onehot_list(range(len(self.reader_list)))
        else:
            self.labels = dbt.numeric_labels_to_onehot_list(labels)
        self.num_classes = len(self.labels)
        #Get names of each file, and create corresponding output_file name
        #These names are only used if a output_fname is not specified 
        for reader in self.reader_list:
            basename = os.path.splitext(reader.name)[0]
            single_output_fname = basename+'.h5'
            self.output_files.append(single_output_fname)
            

        if self.output_fname != None:
            #Create a single training file if self.output_fname was specified
            self.convert_to_single_hdf5(self.reader_list,self.output_fname)
        else:
            self.convert_to_multiple_hdf5(self.reader_list,self.output_files)
        #self.close() #Close each reader after use

    def close(self):
        for reader in self.reader_list:
            #print reader.name,"closed."
            reader.close()
      

    def convert_to_single_hdf5(self,reader_list,output_file,compression=None):
        """
        Convert a list of reader objects into a single hdf5 file. This is the file
        that will be used for training. Random access from a single training file
        should greatly increase training speed.
        """
        num_total_rows = int(np.sum([reader.num_records for reader in reader_list]))
        num_cols = int(self.seq_len*4+self.num_classes)
        
        with h5py.File(output_file,'w') as of:
            of.attrs['seq_len'] = self.seq_len
            of.attrs['num_classes'] = self.num_classes
                
            dset = of.create_dataset('dna',
                                     (num_total_rows,num_cols),
                                    chunks=(64,num_cols),
                                     compression=compression,
                                     maxshape = (1000000,num_cols))


            genidx_file = os.path.splitext(output_file)[0]+'.genidx'
            genidx_handle = open(genidx_file,'w')
            
            offset = 0
            for lbli, reader in enumerate(reader_list):
                reader.open()
                for row in range (reader.num_records):
                    if row%2000==0:
                        print "Wrote",row,"entries to",self.output_fname

                    #Problem if pull 
                    nuc_seqs,rec_ids = reader.pull(1) #Outputs are lists 
                    nuc_onehot = dbt.seq_to_onehot(nuc_seqs[0])
                    label = self.labels[lbli]
                    
                    #Add 1d coded sequence to file
                    dset[offset+row,:] = self.encode_nd_to_1d(nuc_onehot,label)
                    genidx_handle.write(rec_ids[0]+'\n')
                #Add offset of current reader
                offset = offset+reader.num_records
                reader.close()
            print "Wrote", num_total_rows,"to",self.output_fname
            print "Saving annotations to", genidx_file
            genidx_handle.close()

    def convert_to_multiple_hdf5(self,reader_list,output_files,compression):
        """
        Convert each reader into an hdf5 file.
        THIS IS UNTESTED
        
        """
        dset =[]*len(output_files)
        #i is reader index
        for i,hdf in enumerate(output_files):
            with h5py.File(hdf,'w') as of:
                of.attrs['seq_len'] = self.seq_len
                of.attrs['num_classes'] = self.num_classes

                num_rows = reader_list[i].num_records
                num_cols = reader_list[i].seq_len+self.num_classes
                dset[i] = of.create_dataset('dna',
                                        (num_rows,num_cols),
                                        chunks=(64,num_cols),
                                        compression=compression,
                                        maxshape = (1000000,num_cols))

                for row in range (num_rows):
                    if row%2000==0:
                        print "Wrote",j,"entries to",hdf
                    nuc_seq,rec_ids = reader_list[i].pull(1)

                    nuc_onehot = dbt.seq_to_onehot(nuc_seq[0])
                    label = self.labels[i]

                    #Add 1d coded sequence to file
                    dset[i][row,:] = self.encode_nd_to_1d(nuc_onehot,label)
            reader_list[i].close()
                

    def encode_nd_to_1d(self, dna_onehot,label):
        #Flattens a batch to 1-D for saving into an HDF5 file
        dna_onehot_1d = np.ravel(np.squeeze(dna_onehot))
        return np.concatenate((dna_onehot_1d,label),axis=0)

    @staticmethod
    def decode_1d_to_nd(input_1d, shape_list):
        #Decode a 1D matrix into the shapes in shape_list
        return_list = []
        start_ind = 0
        #Slice out return data in their proper shapes
        for shape in shape_list:
            elem_len= np.prod(shape)
            return_list.append(np.reshape(input_1d[start_ind:elem_len+start_ind],shape))
            start_ind += elem_len
        return return_list

    def determine_shapes(self,*argv):
        #Determine the shapes of numpy arrays
        shape_list = []
        for arg in argv:
            shape_list.append(arg.shape)
        return shape_list


class Reader:
    def __init__(self):
        #self.type='BaseReader'
        pass
    
    def pull_as_onehot(self,batch_size):
        #Calls the pull function to retrieve a list of nucleotide sequences
        #Converts the nucleotides to a [batch_size,4,seq_len] onehot numpy array
        nuc_list=self.pull(batch_size)
        
        dna_batch = np.zeros((batch_size,4,self.seq_len))
        #labels_batch = np.ones((batch_size))*label
        for bi,seq in enumerate(nuc_list):
            dna_batch[bi,:,:]=dbt.seq_to_onehot(seq)
        return dna_batch
        
    def pull_dinuc_shuffled(self,batch_size):
        nuc_list,rec_ids=self.pull(batch_size)
        for bi,seq in enumerate(nuc_list):
            #Randomize Ns in seq
            seq = DinucShuffle.replaceN(seq.upper())
            nuc_list[bi] = DinucShuffle.dinuclShuffle(seq)
            rec_ids[bi] = rec_ids[bi]+";shuffled"
        return nuc_list, rec_ids
            
    
    def close(self):
        print "Closed reader for",self.name
        self.parser.close()


        
class CoordsReader(Reader):
    """
    Sequentially reads entries from a bed file of genomic coordinates and
    outputs nucleotides on each pull(num_examples)
    """
    def __init__(self,coord_file,genome_file,chr_sizes_file,seq_len):
        Reader.__init__(self)
        self.coord_file = coord_file
        self.name = self.coord_file
        self.genome_file = genome_file
        self.chr_sizes_file = chr_sizes_file
        self.chr_sizes_dict = dbt.chr_sizes_dict(self.chr_sizes_file)
        self.seq_len = seq_len
        self.index=0
        self.parser = None
        self.num_records = dbt.check_bed_bounds(self.coord_file,self.chr_sizes_dict) 
        
        #self.genome_idx = pysam.FastaFile(self.genome_file)
        #self.open()
        print "CoordsReader object is open. Remember to close by calling obj.close()"
        
    def open(self):
        print "Opening genome file",self.genome_file
        self.genome_idx = pysam.FastaFile(self.genome_file)
        self.parser = open(self.coord_file,'r')

    def close(self):
        self.parser.close()
        
        
    def pull_batch_eval(self,num_examples):
        #This interface method is used to make this class
        #a compatible drop in for InputCollection in certain cases
        return pull(num_examples)
       
    def pull(self,num_examples):
        
        nuc_list=[]
        rec_ids=[]
        #CoordsReader
        """Pull sequence from genome in sequential order"""
        #Correct pull size 
        if self.index+num_examples>self.num_records:
            num_examples = num_examples - ((self.index+num_examples) - self.num_records)
            print "Pulling only",num_examples,"examples at the end of file",self.coord_file

        for i in range(num_examples):
            line= self.parser.readline().strip().split()

            contig,start_str,end_str,name = line[:4]
            contig = str(contig)
            start = int(start_str)
            end= int(end_str)

            #Check start and end bounds
            if (start >= 0) and (end <= int(self.chr_sizes_dict[contig])):
                #Check specified seq_len
                if (end-start)==self.seq_len:
                    nuc_list.append(self.genome_idx.fetch(contig,start,end))
                    rec_id = [name,'\t',contig,':',start_str,'-',end_str]
                    rec_ids.append(''.join(rec_id))
                else:
                    print "Record",(contig,start,end),"does not have seq_len",self.seq_len
                self.index += 1
            else:
                print (contig,start,end),"out of bounds."
                
                
        #print self.index
        return nuc_list,rec_ids
        
        
class FastaReader(Reader):
    """
    Sequentially reads records from a fasta file and outputs nucleotides on each pull
    """
    
    def __init__(self,fasta_file,seq_len):
        Reader.__init__(self)
        if os.path.splitext(fasta_file)[-1] not in ['.fa','.fasta']:
            print "File",fasta_file,"should have \'.fa\' or \'.fasta\' extension."
        self.fasta_file = fasta_file
        self.name = self.fasta_file
        self.seq_len = seq_len

        #self.open()
        #self.parser = SeqIO.parse(self.fasta_file,"fasta")
        self.index = 0

        
        
    def open(self):
        self.num_records = len(SeqIO.to_dict(SeqIO.parse(self.fasta_file,"fasta")))
        
        print "FastaReader object is open. Remember to close by calling obj.close_reader()"
        self.parser = SeqIO.parse(self.fasta_file,"fasta")
        self.index = 0
        
    def pull_batch_eval(self,num_examples):
        #Used to maintain commonality with InputCollection
        return pull(num_examples)
        
    def pull(self,num_examples):
        nuc_list = []
        rec_ids =[]
        #FastaReader
        """Pull fasta records in sequential order"""
        #Correct pull size 
        if self.index+num_examples>self.num_records:
            num_examples = num_examples - ((self.index+num_records) - self.num_records)
            print "Pulling only",num_examples,"examples at the end of file",self.fasta_file
            
        
        for i in range(num_examples):
            nuc_seq = self.parser.next().seq
            rec_id =self.parser.next().id
            if len(nuc_seq) == self.seq_len:
                nuc_list.append(nuc_seq)
                rec_ids.append(rec_id)
            else:
                print("Error. Example",
                      str(i)," sequence length",
                      str(len(nuc_seq)), "does not match",str(self.seq_len) )
                
            self.index +=1
        return nuc_list,rec_id


class DinucShuffleReader(Reader):
    """
    Dinucleotide shuffles the entries of a list of readers
    Takes another reader object as sole input
    """

    def __init__(self,reader_list):
        Reader.__init__(self)
        #Copy all readers in reader_list
        #Note: copy.copy copies will have refs to objects in the copied
        # object, whereas copy.deepcopy will straight copy these objects
        self.reader_list = [copy.copy(reader) for reader in reader_list]
        self.num_records_list = [reader.num_records for reader in self.reader_list]
        self.num_records = np.sum(self.num_records_list)
        self.reader_index = 0 #Index for reader currently being pulled from 
        #self.record_index = 0 # Index of record for current reader
        self.seq_len = self.reader_list[0].seq_len
        self.name = "dinuc_shuffled"
        

        
    def pull(self,batch_size):
        cur_reader = self.reader_list[self.reader_index]
        dinuc_list,rec_ids = cur_reader.pull_dinuc_shuffled(batch_size)
        #self.record_index += batch_size
        #Go to next reader if next batch exceeds num records in cur reader
        if (cur_reader.index)>= cur_reader.num_records:
            self.reader_index += 1
            #self.record_index=0
        
        return dinuc_list,rec_ids

    def open(self):
        for reader in self.reader_list:
            reader.open()
    def close(self):
        for reader in self.reader_list:
            reader.close()
        
        
        
    
class InputCollection:
    def __init__(self,hdf5_file):
        self.filename = hdf5_file
        self.data_key = 'dna'

        with h5py.File(self.filename,'r') as hf:
            self.seq_len=hf.attrs["seq_len"]
            self.num_classes = hf.attrs["num_classes"]
            dna_shape = (4,self.seq_len)
            labels_shape = (1,self.num_classes)
            self.shape_list = [dna_shape,labels_shape]
            #Retrieve num records
            data = hf[self.data_key]
            self.num_records = data.shape[0]

        #Shuffle perm_indices once
        self.perm_indices = np.random.permutation(range(self.num_records))
        
        
        self.train_epoch_tracker = EpochTracker(self.num_records)
        #Epochs for evaluation must be tracked separate to prevent 
        self.eval_epoch_tracker = EpochTracker(self.num_records)

    def set_perm_indices(self,new_indices):
        """
        Used for setting different perm_indices for k-folds
        """
        self.perm_indices = new_indices
        self.num_records = len(self.perm_indices)
        self.train_epoch_tracker = EpochTracker(self.num_records)
        self.eval_epoch_tracker = EpochTracker(self.num_records)
    
        
    def open(self):
        print "Opening HDF5 data file:",self.filename
        self.fhandle = h5py.File(self.filename,'r')
        self.data = self.fhandle[self.data_key]
        
    def close(self):
        "Always remember to call this function when finished!!!"
        self.fhandle.close()

    def _pull_batch(self,batch_size,epoch_tracker):
        #Preallocate
        dna_seq_batch = np.zeros((batch_size,4,self.seq_len))
        labels_batch = np.zeros((batch_size,self.num_classes))
        batch_start = epoch_tracker.cur_index
        batch_end = batch_start + batch_size
        batch_indices = self.perm_indices[batch_start:batch_end]
        
        for bi, pindex in enumerate(batch_indices):
            dna_seq_data,labels_data = ConvertToHdf5.decode_1d_to_nd(self.data[pindex],
                                      self.shape_list)
            dna_seq_batch[bi,:,:] = dna_seq_data
            labels_batch[bi,:] = labels_data
        return dna_seq_batch,labels_batch
        
    def pull_batch_train(self,batch_size):
        dna_seq_batch,labels_batch= self._pull_batch(batch_size,self.train_epoch_tracker)
        epoch_completed = self.train_epoch_tracker.increment(batch_size)
        if epoch_completed:
            #print "Starting on epoch",self.train_epoch_tracker.num_epochs
            self.perm_indices = np.random.permutation(self.perm_indices)
        return dna_seq_batch,labels_batch

    def pull_batch_eval(self,batch_size):
        # Pull a batch for evaluation. This calls the eval epoch tracker
        # And does not shuffle perm indices if end of file reached.
        # This is separate from pull_batch_train to avoid messing with
        # self.epoch_tracker.cur_index every time we need to pull items
        # for evaluation.
        dna_seq_batch,labels_batch= self._pull_batch(batch_size,self.eval_epoch_tracker)
        #Pull a batch for evaluation (calls evaluation epoch tracker)
        epoch_completed = self.eval_epoch_tracker.increment(batch_size)
        if epoch_completed:
            print "Finished evaluating an epoch"
        return dna_seq_batch,labels_batch
    
        

    
class EpochTracker:
    def __init__(self,num_examples):
        #Reminder: this exists as a seperate class to InputCollection
        #because the epoch tracking index need to be tracked separately during training
        # and evaluation
        self.num_examples = num_examples
        self.num_epochs = 0 #The number of epochs that have been passed
        self.cur_index = 0 #The index position on current epoch

    def increment(self,increment_size):
        #Returns true if end of current epoch
        new_index = self.cur_index + increment_size
        #Reset epoch counter if end of current epoch has been reached.
        if ( new_index >= self.num_examples):
            self.num_epochs += 1
            self.cur_index = 0
            #Reshuffle indices
            return True
        else:
            self.cur_index = new_index
            return False

    
