from Bio import SeqIO
import numpy as np
import h5py
from abc import ABCMeta, abstractmethod,abstractproperty
import nucnibble
import dubiotools as dbt
#import pysam
from readers import * #Bed,Fasta,DinucShuffleReader



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

class NucDataRegressMem(BaseNucData):

    '''Just load nucleotide sequences and labels into two equally
       sized lists.
       Labels are quantitative rather than onehot
       This class was originall written to work with DREAM5 data
    '''
    
    def __init__(self):
        self.nuc_seqs = []
        self.labels = []
        self.num_classes=1
        self.num_records = None
        self.seq_len = None

    def add_seq_labels(self,seq_str,label):
        """Add sequences and labels for regression 

        :param seq_str: A string of nucleotides from alphabet AUTGC
        :param label: A float value

        """
        self.nuc_seqs.append(str(seq_str))
        self.labels.append(float(label))


    def get_label_mean_std(self):
        """Used to calculate mean and standard deviation
         Necessary for y feature normalization for regression"""
        if self.num_records == None or self.seq_len == None:
            return None
        
        all_labels = np.zeros((self.num_records))
        for i in range(self.num_records):
            label,_ = self.pull_index_onehot(i)
            all_labels[i] = label

        mean = np.mean(all_labels)
        std = np.std(all_labels)
        return mean,std    

    def split_indices_by_threshold(self,threshold):
        """
        Return indices of items with less than or equal to and
        greater than threshold
        """
        lower_indices = []
        higher_indices = []
        for i in range(self.num_records):
            label,_ = self.pull_index_onehot(i)
            if label>threshold:
                higher_indices.append(i)
            else:
                lower_indices.append(i)
        return lower_indices,higher_indices


    def calc_properties(self):
        self.seq_len = len(self.nuc_seqs[0])
        self.num_records = len(self.labels)
        self.get_label_mean_std()
        assert self.num_records == len(self.nuc_seqs),\
            "Lens of labels list and nuc_seqs list not equal"
        


    def discard_indices_except(self,indices):
        """
        Keep every item with indices passed by user.
        Discard everything else

        """
        new_nuc_seqs = []
        new_labels = []
        #indices.sort() #hmm this breaks encapsulation...
        indices = sorted(indices)
        for i in indices:
            new_nuc_seqs.append(self.nuc_seqs[i])
            new_labels.append(self.labels[i])
        self.nuc_seqs = new_nuc_seqs
        self.labels = new_labels
        self.num_records = len(self.nuc_seqs)
            
        
    def set_classification_threshold(self,threshold):
        self.threshold = threshold
        
    def pull_index_nucs(self,index):
        return self.labels[index],self.nuc_seqs[index]

    def pull_index_onehot(self,index):
        return self.labels[index],dbt.seq_to_onehot(self.nuc_seqs[index])

    
    def close(self):
        """ Dummy method for compatibility with NucHdf5"""

        
    

        
        
                     
class NucDataBedMem(BaseNucData):
    """
    
    Pull nucleotide sequence from bed file by index
    This is largely a wrapper for BedReader
    Given a bed file, genome fasta file, and chromsome sizes file
    
    
    Optional: Generate a dinucleotide shuffled version of the input bed file
              and save as a fasta file
    Optional: Pass two fasta files. One is 
    Note:
    Nucleotide information and labels are stored in two lists.
    All nucleotide data is stored in memory.
    Nucleotide data is stored as a string (not as a nibble array).
    
    """
                
    def __init__(self,
                 bed_file,
                 genome_file,
                 chr_sizes_file,
                 seq_len,
                 skip_first=True,
                 gen_dinuc_shuffle=True,
                 neg_data_reader=None,
                 start_window=None):

        """

        If neg_fasta_file is specified, no dinucleotide shuffled fasta file will be generated
        regardless of whether gen_dinuc_shuffle flag is set to True.

        If start_window is specified, instead of extracting  range [start, end) sequences,
        BedReader will extract [start+start_window[0],start+start_window[1])


        Note: labels are assigned 1 for positive class and 0
        
        """

        self.seq_len = seq_len
        self.start_window=start_window
        if self.start_window:
            print "Using start window {} around start coordinates of .bed file".\
                                       format(start_window)
            
        self.pos_reader = BedReader(bed_file,
                                    genome_file,
                                    chr_sizes_file,
                                    seq_len,
                                    skip_first=True,
                                    start_window=self.start_window)

        self.pos_reader.open()
        #For now, this object must be used with binary classifiers
        self.num_classes =2 


        if neg_data_reader != None:
            self.neg_reader = neg_data_reader
            self.neg_reader.open()
        elif gen_dinuc_shuffle:
            output_fname = os.path.splitext(self.pos_reader.name)[0]+'_dinuc_shuffle.fa'
            if not os.path.exists(output_fname):
                #Dinuc entries from pos_reader and save to fasta file with
                # name output_fname
                dinuc_shuffler = DinucShuffleReader([self.pos_reader])
                dinuc_shuffler.open()
                dinuc_shuffler.save_as_fasta(output_fname)
                dinuc_shuffler.close()
            else:
                print "{0} already found. Will use {0} as dinucleotide shuffled dataset".\
                                                                    format(output_fname)
            #Create fasta file to record entries in DinucShuffleReader
            self.neg_reader = FastaReader(output_fname,self.seq_len)
            self.neg_reader.open()
        else:
            print "Must specify either gen_dinuc_shuffle=True or explicitly provide a neg_data_reader"

            


        all_pulls = []

        ###Read all sequences into memory

        #Must have negative reader
        assert self.neg_reader
        readers = [self.pos_reader,self.neg_reader]


        for reader in readers:
            while True:
                cur_pull =reader.pull(1)
                if cur_pull[0] != [] and cur_pull[1] != []:
                    all_pulls.append(cur_pull)
                else:
                    break

        

        print "Num positive examples pulled: {}".format(self.pos_reader.num_pulled)
        print "Num negative examples pulled: {}".format(self.neg_reader.num_pulled)
        assert self.pos_reader.num_pulled > 0
        assert self.neg_reader.num_pulled > 0

        '''
        for i in zip(*all_pulls)[0]:
            try:
                print i[0][0]
            except IndexError:
                print i

        '''
        self.nuc_seqs = [i[0] for i in zip(*all_pulls)[0]]
            
        self.labels = [1]*self.pos_reader.num_pulled + [0]*self.neg_reader.num_pulled
        self.num_records = len(self.nuc_seqs)
        
        self.pos_reader.close()
        self.neg_reader.close()

        
    def pull_index_nucs(self,index):
        return self.labels[index],self.nuc_seqs[index]

    def pull_index_onehot(self,index):
        return self.labels[index],dbt.seq_to_onehot(self.nuc_seqs[index])

    
    def close(self):
        self.bed_reader.close()

                
class NucDataFastaMem(BaseNucData):
    """
    Load a set of fasta files into memory. Labels should be of type
        classification

        fasta_file[0] gets label 0 (negative class for binary classifier)
        fasta_file[1] gets label 1
    """
    
    def __init__(self, fasta_files, seq_len):

        if len(fasta_files)!=2:
            "Error. This object can currently only take a maximum of two fasta files"
            return None
        self.num_classes =2 
        self.fasta_files = fasta_files
        self.num_records = 0
        self.seq_len = seq_len
        #self.nibble_seq_len = int(seq_len//2+seq_len%2 +4)
        ##List of every fasta seq object
        self.seq_parser_list = []

        #For now, this object must be used with binary classifiers


        
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
            lower_bound = upper_bound
            #first_rec = seq_parser.next().seq
            #self.seq_len = len(first_rec)
            #first_nib_seq = nucnibble.nuc_to_uint8_numpy(first_rec)
            #self.nibble_seq_len = len(first_nib_seq)

        #Preallocate array (this makes things memory efficient)
        #self.nib_array = np.zeros((self.num_records,self.nibble_seq_len),dtype='float32')
        print self.bounds
          
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


