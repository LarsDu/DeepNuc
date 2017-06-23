import sys
import os
from collections import defaultdict
import pysam

import numpy as np

#Go one directory up and append to path to access the deepdna package
#The following statement is equiv to sys.path.append("../")
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)))

import gflags

FLAGS = gflags.FLAGS
#base_dir='../../Genomes'
#def_cds_bed_file = base_dir + os.sep+ 'CDS/ce10_coding_exons_refseq.bed'
#def_genome = base_dir + os.sep+ '
#def_chrom_sizes = base_dir + os.sep+

gflags.DEFINE_string('cds_bed_file', '' ,""" Tab delimited bed style file """)
gflags.DEFINE_string('genome', '', """ Genome fasta file """)
#gflags.DEFINE_string('chrom_sizes', '', """ Chromsome sizes file  """)
gflags.DEFINE_integer('seq_len', 600, """ Sequence length to extract  """)
gflags.DEFINE_integer('num_records',5000,""" Number of records to pull at random """)
gflags.DEFINE_string('output_dir', '.', """ Directory to save output file """)

def main(argv):
   
    #Parse gflags
    try:
        py_file = FLAGS(argv)  # parse flags
    except gflags.FlagsError, e:
        print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
        sys.exit(1)


    rand_cds_to_fasta(FLAGS.genome,
                      FLAGS.cds_bed_file,
                      FLAGS.num_records,
                      FLAGS.seq_len,
                      FLAGS.output_dir)



def rand_cds_to_fasta(genome, cds_bed_file, num_records, seq_len,output_dir):
    output_fasta = os.path.splitext(os.path.basename(cds_bed_file))[0]
    output_fasta = output_dir+os.sep+output_fasta+'_{}recs.{}'.format(num_records,'fa')
    genome_idx = pysam.FastaFile(genome)
    print cds_bed_file
    tdict = transcript_dict(cds_bed_file)
    tdict_keys = tdict.keys()
    num_transcripts = len(tdict_keys)
    print "Bed file {} contains {} transcripts".format( cds_bed_file,num_transcripts)
    perm_ind = np.random.permutation(range(num_transcripts))

    #list of lists of tuples
    #each list corresponds to a transcript
    #Select either 1.5* number of desired records or the number of transcripts
    approx_num_records = min(int(num_records*1.5),len(tdict_keys)) 
    tlists = [tdict[tdict_keys[i]] for i in perm_ind[0:approx_num_records]] 

    counter = 0
    with open(output_fasta,'w') as of:
        for i,tscript in enumerate(tlists):
            cur_transcript = ''
            cur_contig = ''
            tscript_start=None
            for exon in tscript:
                contig,start,end = exon
                cur_contig = contig
                if not tscript_start:
                    tscript_start = int(start)
                cur_transcript += genome_idx.fetch(contig,int(start),int(end))
            cur_seq_len = len(cur_transcript)

            #If the transcript is long enough to contain at least 2 seq_lens,
            #extract 2 sequences
            if cur_seq_len > 2*seq_len:
                tscript = [cur_transcript[0:cur_seq_len//2],cur_transcript[cur_seq_len//2:]]
            else:
                tscript = [cur_transcript]

            for j,each_script in enumerate(tscript):
                rand_upper = len(each_script)-seq_len
                if rand_upper > 0:
                    rand_start = np.random.randint(0,rand_upper)
                    rand_end = rand_start+seq_len
                    abs_start = tscript_start+rand_start
                    abs_end = tscript_start+rand_end
                    of.write('>'+tdict_keys[i]+' '+str(j)+' '+
                             cur_contig+':'+str(abs_start)+'-'+str(abs_end)+'\n')
                    of.write(each_script[rand_start:rand_end]+'\n')
                    counter += 1

                
            if counter >= num_records:
                break

    print "Wrote {} records to {}".format(counter,output_fasta)
    #TODO: make sure output file is in the correct directory.
    
    
def transcript_dict(cds_bed_file):
    tdict = defaultdict(list)

    with open(cds_bed_file,'r') as bf:
        bf.readline() #Ignore header
       
        for line in bf:
            contig,start,end,name,_,strand = line.split()
            transcript = name.split('_')
            transcript_id = transcript[0]+'_'+transcript[1]
            tdict[transcript_id].append((contig,start,end))  


            
    return tdict
if __name__ == "__main__":
    main(sys.argv)
