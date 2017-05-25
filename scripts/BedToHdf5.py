import gflags
import sys
import os

#Go one directory up and append to path to access the deepdna package
#The following statement is equiv to sys.path.append("../")
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)))

from deepnuc.converters import ConvertToHdf5
from deepnuc.readers import BedReader 
from deepnuc.readers import DinucShuffleReader



"""
Converts a list of coordinate files (typically in bed format) to a single
Hdf5 file with labels corresponding to each file.

"""
FLAGS = gflags.FLAGS

gflags.DEFINE_string('file_list','',"""A comma delimited list of input coordinate files""")
gflags.DEFINE_string('genome_file','',"""A genome reference file in fasta format""")
gflags.DEFINE_string('chrom_sizes_file','',"""A listing of chromosome sizes for genome_file""")
gflags.DEFINE_integer('seq_len',600,"""Nucleotide length of all training examples""") 
gflags.DEFINE_string('output_fname','',"""Output HDF5 file. If this
                                                    option is left '', an HDF5 file
                                                    will be generated for each input file""")
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

    
    reader_list = [BedReader(coords_file,
                                FLAGS.genome_file,
                                FLAGS.chrom_sizes_file,
                                FLAGS.seq_len) for coords_file in file_list]

    dinuc_reader = []
    if FLAGS.dinuc_shuffle == True:
        dinuc_reader = [DinucShuffleReader(reader_list)]
        

    if FLAGS.output_fname=='':
        fbase = os.path.splitext(reader_list[0].name)[0]
        output_fname = fbase+'.h5'
        print "No output name specified"
        print "Naming file",output_fname
    else:
        output_fname = FLAGS.output_fname
        
    converter = ConvertToHdf5(dinuc_reader+reader_list,
                          output_fname = output_fname)
                          


if __name__=='__main__':
    main(sys.argv)
