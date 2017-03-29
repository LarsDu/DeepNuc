import sys
import os
import numpy as np
import copy

import dinucshuffle
from Bio import SeqIO
import dubiotools as dbt

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
            seq = dinucshuffle.replaceN(seq.upper())
            nuc_list[bi] = dinucshuffle.dinuclShuffle(seq)
            rec_ids[bi] = rec_ids[bi]+";shuffled"
        return nuc_list, rec_ids
            
    
    def close(self):
        print "Closed reader for",self.name
        self.parser.close()


        
class BedReader(Reader):
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
        #print "BedReader object is open. Remember to close by calling obj.close()"
        
    def open(self):
        print "Opening BedFile",self.coord_file,"with",self.num_records,"records"
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
        #BedReader
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
        self.num_records = len(SeqIO.index(self.fasta_file,"fasta"))
        self.index = 0
        #self.open()
        
        
    def open(self):
        #self.num_records = len(SeqIO.to_dict(SeqIO.parse(self.fasta_file,"fasta")))
        print "Opening FastaReader with",self.num_records,"records"
        self.parser = SeqIO.parse(self.fasta_file,"fasta")
        
        
    def pull_batch_eval(self,num_examples):
        #Used to maintain commonality with InputCollection
        return pull(num_examples)
        
    def pull(self,num_examples):
        if self.index >= self.num_records:
            print "Error! No more records to pull"
            print "Num records:",self.num_records
            print "Index:",self.index 
        nuc_list = []
        rec_ids =[]
        #FastaReader
        """Pull fasta records in sequential order"""
        #Correct pull size 
        if self.index+num_examples>self.num_records:
            num_examples = num_examples - ((self.index+num_records) - self.num_records)
            print "Pulling only",num_examples,"examples at the end of file",self.fasta_file

        for i in range(num_examples):
            try:
                seq_obj = self.parser.next()
                nuc_seq = seq_obj.seq
                rec_id = seq_obj.id
            except StopIteration:
                print "Failure in FastaReader pull at", self.index
            if len(nuc_seq) == self.seq_len:
                nuc_list.append(nuc_seq)
                rec_ids.append(rec_id)
            else:
                print("Error. Example",
                      str(i)," sequence length",
                      str(len(nuc_seq)), "does not match",str(self.seq_len) )
                
            self.index +=1
        return nuc_list,rec_ids


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
        self.seq_len = self.reader_list[0].seq_len
        self.name = "dinuc_shuffled"
        

        
    def pull(self,batch_size):
        cur_reader = self.reader_list[self.reader_index]
        dinuc_list,rec_ids = cur_reader.pull_dinuc_shuffled(batch_size)
        #self.record_index += batch_size
        #Go to next reader if next batch exceeds num records in cur reader
        if (cur_reader.index)>= cur_reader.num_records:
            self.reader_index += 1
        return dinuc_list,rec_ids

    def open(self):
        print "Opening DinucReader with",self.num_records,"records"
        for reader in self.reader_list:
            reader.open()
    def close(self):
        for reader in self.reader_list:
            reader.close()
        
