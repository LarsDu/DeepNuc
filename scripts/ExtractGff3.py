import gflags
import sys
import os


#Go one directory up and append to path to access the deepdna package
#The following statement is equiv to sys.path.append("../")
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)))

FLAGS = gflags.FLAGS
gflags.DEFINE_string('genome_file','',"""A genome reference file in fasta format""")
gflags.DEFINE_string('gff3_file','',"""GFF3 annotation file""")
gflags.DEFINE_string('feature','',"""The feature to extract from a given genome""")
gflags.DEFINE_string('output','',"""The output format. Can be""")


def main(argv):
        #Parse gflags
    try:
        py_file = FLAGS(argv)  # parse flags
    except gflags.FlagsError, e:
        print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
        sys.exit(1)

    test_parser = Gff3Parser(FLAGS.gff3_file,FLAGS.feature)
    for _ in range(20):
        print test_parser.next()['start']

if __name__ == "__main__":
    main()
