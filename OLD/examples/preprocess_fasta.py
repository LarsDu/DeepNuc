import DinucShuffle as dinuc
from Bio import SeqIO
from Bio.Alphabet import IUPAC
import os
import sys
from random import shuffle
import math 
import gflags
from __future__ import print_function 



FLAGS = gflags.FLAGS


gflags.DEFINE_float('train_frac',.75,"""Training fraction""")
gflags.DEFINE_float('validation_frac',0,"""Validation fraction""")
gflags.DEFINE_float('test_frac',.25,"""Test fraction""")



"""
1. Takes a fasta file or multiple fasta files as input from commandline
2. Generates a dinucleotide shuffled version of each fasta file
3. Splits each fasta file into training, validation, and test sets
"""

def main(argv):
    
    for i,file in enumerate(argv):
        if i>0:
            fname,fext = os.path.splitext(file)
            output_file = open(fname+"_dinuc_shuffled"+fext,'rU')
            dinuc_shuffle(file,output_file)
            output_file.close()
        



def dinuc_shuffle(fname,output_file):
    seq_parser = SeqIO.parse(fname,"fasta")
    for record in seq_parser:
        #Writes dinucleotide shuffled version of each input sequence
        #to stddout
        A= dinuc.generate_sequences(seq_parser,1,output_file)
    seq_parser.close()

def read_files(fname_list):
    if type(fname_list) is not list:
        print "Converting to list"
        fname_list = [fname_list]

    for file in fname_list:
        print 'Opening file',file
        handle = open(file,"rU")
        #Read whole file as list
        records = list(SeqIO.parse(handle,"fasta"))
        handle.close()
        #Shuffle records in place
        shuffle(records)
        #Extract appropriate number of elements for
        num_records = len(records)
        print ('File ',file,' contains ', len(records), ' records.')
        num_train = int(math.floor(FLAGS.train_frac * num_records))
        num_validation = int(math.floor(FLAGS.validation_frac*num_records))
        num_test = int(math.floor(FLAGS.test_frac*num_records))

        print "Number of training examples:\t",num_train
        print "Number of validation examples:\t", num_validation
        print "Number of test examples:\t", num_test


        
        if num_train:
            _write_file(file,'train',records[0:num_train])
        if num_test:
            _write_file(file,'test',records[num_train:num_train+num_validation])
        if num_validation:
            _write_file(file,'validation', records[num_train+num_validation:])




def _write_file(input_fname,name_extension,seqio_records_list):
    fname, file_extension = os.path.splitext(input_fname)
    output_fname = fname+'_'+name_extension+file_extension
    output_handle = open(output_fname,"w")
    SeqIO.write(seqio_records_list,output_handle,"fasta")
    output_handle.close()


    
if __name__ == '__main__':
    main(sys.argv)
