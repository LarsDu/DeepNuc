import gflags
import sys
import os

#Go one directory up and append to path to access the deepnuc package
#The following statement is equiv to sys.path.append("../")
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)))

from deepnuc.converters import ConvertToHdf5
from deepnuc.readers import FastaReader 
from deepnuc.readers import DinucShuffleReader



"""
Converts fasta files to a single Hdf5 file with labels
corresponding to each file.
"""

FLAGS = gflags.FLAGS

gflags.DEFINE_string('file_list','',"""A comma delimited list of input fasta files""")
gflags.DEFINE_integer('seq_len',600,"""Nucleotide length of all training examples""") 
gflags.DEFINE_string('output_fname','default.h5',"""Output HDF5 file.""")
gflags.DEFINE_boolean('dinuc_shuffle',False,"""Set true to generate a dinucleotide
                                             shuffled version of each input reader.""")



def main(argv):
    #Parse gflags
    try:
        py_file = FLAGS(argv)  # parse flags
    except gflags.FlagsError, e:
        print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
        sys.exit(1)

        
    file_list = FLAGS.file_list.strip().split(',')
    print "Input files:",file_list
    reader_list = [FastaReader(fasta_file,FLAGS.seq_len) for fasta_file in file_list]

    dinuc_reader = []
    if FLAGS.dinuc_shuffle == True:
        dinuc_reader = [DinucShuffleReader(reader_list)]


        
    print "Converting to",FLAGS.output_fname
   
    conv_obj=ConvertToHdf5(dinuc_reader+reader_list,
                          output_fname = FLAGS.output_fname)


        
if __name__=='__main__':
    main(sys.argv)



