from Bio import SeqIO
import numpy as np
import h5py
from abc import ABCMeta, abstractmethod,abstractproperty
import nucnibble
import dubiotools as dbt



class BaseNucData():
    """
    Abstract class for holding nuc data
    This is a like a contract for subclasses to 
    contain the properties and methods listed below
    """

    __metaclass__= ABCMeta
    
    
    @abstractmethod
    def pull_index_nucs(self,index):
        raise NotImpletmentedError()

    @abstractmethod
    def pull_index_onehot(self,index):
        raise NotImpletmentedError()

    @abstractmethod
    def close(self):
        raise NotImpletmentedError()

    '''
    @abstractproperty
    def num_records(self,new_value):
        return new_value

   
    @abstractproperty
    def seq_len(self,new_value):
        return new_value
    '''

class NucDataRegression(BaseNucData):

    def __init__(self):
        self.nuc_seqs = []
        self.labels = []
        self.num_classes=1

    def add_seq_labels(self,seq_str,label):
        """Add sequences and labels for regression 

        :param seq_str: A string of nucleotides from alphabet AUTGC
        :param label: A float value

        """
        self.nuc_seqs.append(str(seq_str))
        self.labels.append(float(label))

  
        
    
class NucMemoryRegression(NucDataRegression):
    '''Just load nucleotide sequences and labels into two equally
       sized lists.
       Labels are quantitative rather than onehot
       This class was originall written to work with DREAM5 data
    '''

    def __init__():
        super(NucDataRegression, self).__init__()
    
        
    def calc_properties(self):
        self.seq_len = len(self.nuc_seqs[0])
        self.num_records = len(self.labels)
        assert self.num_records == len(self.nuc_seqs),\
            "Lens of labels list and nuc_seqs list not equal"
        

    def pull_index_nucs(self,index):
        return self.labels[index],self.nuc_seqs[index]

    def pull_index_onehot(self,index):
        return self.labels[index],dbt.seq_to_onehot(self.nuc_seqs[index])

    
    def close(self):
        """ Dummy method for compatibility with NucHdf5"""
                     
        
        
        
class NucFastaMemory(BaseNucData):
    """ Load a set of fasta files into memory. Labels should be of type
        classification """
    
    def __init__(self, fasta_files, seq_len):
        self.fasta_files = fasta_files
        self.num_records = 0
        self.seq_len = seq_len
        #self.nibble_seq_len = int(seq_len//2+seq_len%2 +4)
        ##List of every fasta seq object
        self.seq_parser_list = []

        #For now, this object must be used with binary classifiers
        self.num_classes =2 
        
        #List holding tuples for index ranges corresponding to different labels
        #Example: Fasta1.fa has 3 records. Fasta2.fa has 4. self.bounds =[(0,3),(3,7)]
        self.bounds = []
        
        #Populate seq_parser list (This is quite memory intensive)
        lower_bound = 0
        for label,fname in enumerate(fasta_files):
            seq_parser = list(SeqIO.parse(fname,"fasta"))
            self.seq_parser_list.extend(seq_parser)
            num_recs = len(seq_parser)
            self.num_records += num_recs
            upper_bound = lower_bound+num_recs
            self.bounds.append((lower_bound,upper_bound))
            #first_rec = seq_parser.next().seq
            #self.seq_len = len(first_rec)
            #first_nib_seq = nucnibble.nuc_to_uint8_numpy(first_rec)
            #self.nibble_seq_len = len(first_nib_seq)

        #Preallocate array (this makes things memory efficient)
        #self.nib_array = np.zeros((self.num_records,self.nibble_seq_len),dtype='float32')
        
  
    def pull_index_nucs(self,index):
        numeric_label = self.label_from_index(index)
        nuc_seq = str(self.seq_parser_list[index].seq)
        return numeric_label,nuc_seq

    def pull_index_onehot(self,index):
        numeric_label = self.label_from_index(index)
        iseq = self.seq_parser_list[index].seq
        nuc_onehot = dbt.seq_to_onehot(iseq) #(outputs nx4)
        return numeric_label,nuc_onehot

    def label_from_index(self,index):
        for i,bound in enumerate(self.bounds):
            if index >= bound[0] and index < bound[1]:
                return i
        
    
    def close(self):
        """ Dummy method for compatibility with NucHdf5"""
        pass


        
class NucHdf5(BaseNucData):
   
    def __init__(self,hdf5_file):
        self.filename = hdf5_file
        self.data_key = 'dna'
        self.fhandle = None #Set default state
        self.open() #Opens file, sets self.num_records
           
    def open(self):
        print "Opening HDF5 data file:",self.filename
        self.fhandle = h5py.File(self.filename,'r')
        self.data = self.fhandle[self.data_key]
        self.seq_len = self.fhandle.attrs["seq_len"]
        self.num_classes = self.fhandle.attrs["num_classes"]
        self.num_records = self.data.shape[0]
        self.onehot_labels_list = dbt.numeric_labels_to_onehot_list(range(self.num_classes))

    def pull_index_nucs(self,index):
        if self.fhandle == None:
            print "Error, NucHdf5 must have fhandle specified!"
            print "Please open()"
        numeric_label = self.data[index,0] #single digit
        #Decode uint8 values to a nucleotide string
        nuc_string = nucnibble.uint8_numpy_to_nucs(self.data[index,1:])
        return numeric_label,nuc_string

    def pull_index_onehot(self,index):
        numeric_label = self.data[index,0]
        onehot_label = self.onehot_labels_list[numeric_label]
        #Decode uint8 values to a 4 x seq_len numpy one-hot array
        nuc_onehot = nucnibble.uint8_numpy_to_onehot_nucs(self.data[index,1:]).T
        return numeric_label,nuc_onehot
            
    def close(self):
        "Always remember to call this function when finished!!!"
        self.fhandle.close()


