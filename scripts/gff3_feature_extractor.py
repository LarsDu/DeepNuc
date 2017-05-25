import gflags
import sys
import os
from collections import defaultdict

FLAGS = gflags.FLAGS
#gflags.DEFINE_string('genome_file','',"""A genome reference file in fasta format""")
gflags.DEFINE_string('gff3_file','annotations/test.gff3',"""GFF3 annotation file""")
#gflags.DEFINE_string('feature','',"""The feature to extract from a given genome""")
gflags.DEFINE_string('output','',"""The output format. Can be""")


def main(argv):
        #Parse gflags
    try:
        py_file = FLAGS(argv)  # parse flags
    except gflags.FlagsError, e:
        print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
        sys.exit(1)

    #test_iter = Gff3Iter(FLAGS.gff3_file)
    #for _ in range(4):
    #    print test_iter.next()['source']

    #for item in test_iter:
    #    print item
    gparser = Gff3Parser(FLAGS.gff3_file)
    gparser.enumerate_gff3_keys(['seqid','source','type'])
    #gparser.enumerate_gff3_key('seqid')
    #gparser.enumerate_gff3_key('source')
    #gparser.enumerate_gff3_key('type')


class Gff3Parser:
    def __init__(self,filename):
        self.gff_keys = ["seqid","source","type","start",
                         "end","score","phase","attributes"]
    
        self.attributes = ["ID", "Name", "Alias", "Parent",
                           "Target", "Gap", "Derives_from",
                                "Note","Dbxref","Ontology_term","Is_circular"]
        self.filename = filename
        self.iterator = Gff3Iter(self.filename)


    
    def enumerate_gff3_keys(self,keys):
        if not type(keys) == list:
            keys = list(keys)
    
        d = defaultdict(set)
        for item in self.iterator:
            for key in keys:
                d[key].add(item[key])
        self.reset_iterator()

        #Print out
        for key in keys:
            print "\nKey:\t{}".format(key)
            print d[key]

        return d

    '''
    def enumerate_gff3_key(self,key):
        val_set = set()
        for item in self.iterator:
            val_set.add(item[key])
        self.reset_iterator()
        print val_set
        return val_set
    '''

    
    def print_dict_as_line(self,item):
        line = [item[key] for key in self.gff_keys]
        '\t'.join(line)
        

    def get_lines_with_value(key,value):
        """ Print all lines with a given key and value"""
        for item in self.iterator:
            if item[key] == value:
                print print_dict_as_line(item)

        self.reset_iterator()
        
    
    def get_values(self,key):
        """Get all values corresponding to a particular key"""
        dict_list = []
        if key in self.gff_keys:
            for item in self.iterator:
                dict_list.append(item[key])
        self.reset_iterator()
        return dict_list

    def reset_iterator(self):
        self.iterator.close()
        self.iterator = Gff3Iter(self.filename)



class Gff3Iter(object):
    def __init__(self,filename):

        self.gff_keys = ["seqid","source","type","start",
                         "end","score","phase","attributes"]
    
        
        self.filename = filename
        self.cur_line=None
        self.counter = 0
        self.open()

    def open(self):
        print "Opening",self.filename
        self.fhandle = open(self.filename,'r')

    def close(self):
        if self.fhandle:
            self.fhandle.close()

    def __iter__(self):
        return self


    def next(self):
        self.cur_line = self.fhandle.readline()
        if self.cur_line == '':
            raise StopIteration

        if self.cur_line.startswith("#"):
            #Recursive call will keep going until an uncommented line is found
            return self.next()
        else:
            vals = self.cur_line.strip('\n').split('\t')
            self.counter += 1
            return dict(zip(self.gff_keys,vals))
        
      
      
    
    
if __name__ == "__main__":
    main(sys.argv)           
