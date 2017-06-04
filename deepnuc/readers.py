import sys
import os
import numpy as np
import copy
import pysam

import dinucshuffle
from Bio import SeqIO
import dubiotools as dbt

class Reader(object):
    def __init__(self):
        #self.type='BaseReader'
        pass


    def pull(self):
        return ([],[])
        
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
            seq = dinucshuffle.replaceN(seq.upper())
            nuc_list[bi] = dinucshuffle.dinuclShuffle(seq)
            rec_ids[bi] = rec_ids[bi]+"; dinucleotide_shuffled"
        return nuc_list, rec_ids
            
    def get_num_records(self):
        #Very expensive call. Must call pull for entire dataset
        self.num_records = 0
        while True:
            nuc_list,rec_ids = self.pull(1)
            if nuc_list == [] or rec_ids == []:
                break
            else:
                self.num_records += 1
        #reset pull counter
        self.num_pulled = 0
        
    def close(self):
        print "Closed reader for",self.name
        self.parser.close()


        
class BedReader(Reader):
    """
    Sequentially reads entries from a bed file of genomic coordinates and
    outputs nucleotides on each pull(num_examples)

    if start_window is defined, instead of reading coordinates
    exactly, BedReader will read a window relative to the start coordinate
    """
    def __init__(self,
                 coord_file,
                 genome_file,
                 chr_sizes_file,
                 seq_len,
                 skip_first=True,
                 start_window=None,
                 pull_limit = -1,
                 filter_by_len = False
                 ):
        """Read a bed file with many options for preprocessing

        :param coord_file: '.bed' file
        :param genome_file: '.fa' file with whole genome 
        :param chr_sizes_file: chromosome sizes file corresponding to aforementioned genome 
        :param seq_len: The sequence length to extract from the start coord.
                        This also represents the maximum length that can be pulled.
                        bed file lines shorter than this value will not be pulled. 
        :param skip_first: Skip the first line of the file (the header line)
        :param start_window: A tuple (ie: (-100,100). If specified, extract this
                             range from around the start window instead of using
                             both start and end 
        :param pull_limit: If specified, only extract the specified number of lines
                             (excluding header if skip_first is set to True).
                              Pull all if set to -1
        :param filter_by_len: If True, only pull from lines where
                              end-start > self.seq_len
        :returns: a BedReader object
        :rtype: BedReader type object
        """
        
        Reader.__init__(self)
        self.coord_file = coord_file
        self.name = self.coord_file
        self.genome_file = genome_file
        self.chr_sizes_file = chr_sizes_file
        self.chr_sizes_dict = dbt.chr_sizes_dict(self.chr_sizes_file)
        self.seq_len = seq_len
        self.parser = None
        self.num_pulled= 0
        #self.num_records = dbt.check_bed_bounds(self.coord_file,self.chr_sizes_dict)
        self.skip_first = skip_first
        self.start_window = start_window
        self.pull_limit = pull_limit
        self.filter_by_len = filter_by_len
        
        
    def open(self):
        print "Opening BedFile",self.coord_file
        print "Opening genome file",self.genome_file
        self.genome_idx = pysam.FastaFile(self.genome_file)
        self.parser = open(self.coord_file,'r')
        if self.skip_first==True:
            self.parser.readline()

    def close(self):
        self.parser.close()
        
    '''
    def pull_batch_eval(self,num_examples):
        #This interface method is used to make this class
        #a compatible drop in for InputCollection in certain cases
        return pull(num_examples)
    '''
        
    def pull(self,num_examples):
        if self.pull_limit > -1 and self.num_pulled > self.pull_limit:
            return [],[]
        #Returns empty lists on failure
        nuc_list=[]
        rec_ids=[]
        #BedReader
        """Pull sequence from genome in sequential order"""

        for i in range(num_examples):
            line= self.parser.readline().strip().split()
            if line != []:
                contig,start_str,end_str = line[:3]
                contig = str(contig)
                real_start = int(start_str)
                start = real_start
                end= int(end_str)
                real_len = end-real_start 
                if self.filter_by_len and real_len < self.seq_len:
                    #Recursively call this method until an acceptably long sequence is
                    #pulled.
                    #self.num_pulled will only get pulled if successful.
                    return self.pull(num_examples)
                
                if self.start_window:
                    #Use a window around the start position instead
                    start = real_start+self.start_window[0]
                    end = real_start+self.start_window[1]
                
                
                
                #Check start and end bounds
                if (start >= 0) and (end <= int(self.chr_sizes_dict[contig])):
                    #Check specified seq_len
                    if (end-start)==self.seq_len:
                        seq= self.genome_idx.fetch(contig,start,end)
                        #Check pulled sequence
                        if len(seq) == self.seq_len:
                            nuc_list.append(seq)
                            rec_id = [contig,':',start_str,'-',end_str]
                            rec_ids.append(''.join(rec_id))
                            self.num_pulled += 1
                        else:
                            print "Error! record {}:{}-{} did not yield correct sequence length {}".\
                                             format(contig,start_str,end_str,self.seq_len)
                            print "on pysam pull."

                    else:
                        print "Record",(contig,start,end),"does not have seq_len",self.seq_len
                else:
                    print (contig,start,end),"out of bounds."
                
        actual_num_examples = len(nuc_list)
        if actual_num_examples != num_examples:
            print "Reached end of file and only pulling {} from file {}".\
                 format(actual_num_examples,self.coord_file)
            print "Pulled {} records from file {}".\
                       format(self.num_pulled,self.coord_file)

        return nuc_list,rec_ids
        
        
class FastaReader(Reader):
    """
    Sequentially reads records from a fasta file and outputs nucleotides on each pull
    """
    
    def __init__(self,fasta_file,seq_len,pull_limit=-1):
        Reader.__init__(self)
        if os.path.splitext(fasta_file)[-1] not in ['.fa','.fasta']:
            print "File",fasta_file,"should have \'.fa\' or \'.fasta\' extension."
        self.fasta_file = fasta_file
        self.name = self.fasta_file
        self.seq_len = seq_len
        #self.num_records = len(SeqIO.index(self.fasta_file,"fasta"))
        self.num_pulled = 0 #Determine number of records by pulls
        self.pull_limit = pull_limit
        #self.open()
        
        
    def open(self):
        #self.num_records = len(SeqIO.to_dict(SeqIO.parse(self.fasta_file,"fasta")))
        print "Opening FastaReader {}".format(self.fasta_file)
        self.parser = SeqIO.parse(self.fasta_file,"fasta")
        
        
    def pull_batch_eval(self,num_examples):
        #Used to maintain commonality with InputCollection
        return pull(num_examples)
        
    def pull(self,num_examples):
        if self.pull_limit > -1 and self.num_pulled > self.pull_limit:
            return [],[]
   
        #FastaReader
        #Returns empty lists on failure
        nuc_list = []
        rec_ids =[]

        """Pull fasta records in sequential order"""
                
        for i in range(num_examples):
            try:
                seq_obj = self.parser.next()
                nuc_seq = str(seq_obj.seq)
                rec_id = seq_obj.id

                if len(nuc_seq) == self.seq_len:
                    nuc_list.append(nuc_seq)
                    rec_ids.append(rec_id)
                    self.num_pulled += 1
                else:
                    print("Error. Example",
                        str(i)," sequence length does not match",str(self.seq_len) )
            
            except StopIteration:
                print "Failure in FastaReader pull at", self.num_pulled
                

        actual_num_examples = len(nuc_list)
        if actual_num_examples != num_examples:
            print "Reached end of file and only pulling {} from file {}".\
                                   format(actual_num_examples,self.fasta_file)

            print "Pulled {} records from fasta file {}".\
                       format(self.num_pulled,self.fasta_file)

        return nuc_list,rec_ids


class DinucShuffleReader(Reader):
    """
    Dinucleotide shuffles the entries of a list of readers
    Takes another reader object as sole input

    Note: every pull will perform a unique shuffling operation.
    To cache pulls, save the output of this reader as a Fasta file
    """

    def __init__(self,reader_list):
        Reader.__init__(self)
        #Copy all readers in reader_list
        #Note: copy.copy copies will have refs to objects in the copied
        # object, whereas copy.deepcopy will straight copy these objects
        self.reader_list = [copy.copy(reader) for reader in reader_list]
        
        self.reader_index = 0 #Index for reader currently being pulled from 
        self.seq_len = self.reader_list[0].seq_len
        self.name = "dinuc_shuffled"
        self.num_pulled = 0
        
    def save_as_fasta(self,output_file):
        print "Saving dinucleotide shuffled entries in file",output_file
        self.reader_index=0
        with open(output_file,'w') as of:
            while(True):
                dinuc_list,rec_ids = self.pull(1)
                if not (dinuc_list == [] or dinuc_list == None or
                        rec_ids == [] or rec_ids == None):
                    of.write(">{}\n".format(rec_ids[0]))
                    of.write(dinuc_list[0]+"\n")
                else:
                    break
        self.reader_index=0
        
    def pull(self,batch_size):
        #Returns empty lists on final pull
        cur_reader = self.reader_list[self.reader_index]
        dinuc_list,rec_ids = cur_reader.pull_dinuc_shuffled(batch_size)
        #Go to next reader if pull fails
        if dinuc_list == [] or rec_ids == []:
            self.reader_index += 1
        else:
            self.num_pulled += len(dinuc_list)
        return dinuc_list,rec_ids

    def open(self):
        print "Opening DinucReader"
        for reader in self.reader_list:
            reader.open()
    def close(self):
        for reader in self.reader_list:
            reader.close()
        
