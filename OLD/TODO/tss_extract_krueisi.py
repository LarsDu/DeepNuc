import csv
import gflags
from Bio import SeqIO
import sys
import os

FLAGS = gflags.FLAGS

gflags.DEFINE_string('kruesi_file',
                     'Kruesi_2013_data/elife-00808-fig1-data2-v1-download.csv',
                     """ The TSS file from Kruesi """)
gflags.DEFINE_string('genome_fasta',
            os.path.expanduser('~')+os.sep+'Genomes/link_worm_genomes/WS230/c_elegans.WS230.genomic.fa', """ Genome fasta file """)


def main(argv):
    """
    Divide a fasta file into training set, cross_validation set, and
    test set, attaching appropriate labels to resulting files.

    """

    #Parse gflags
    try:
        py_file = FLAGS(argv)  # parse flags
    except gflags.FlagsError, e:
        print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
        sys.exit(1)
    #if FLAGS.debug:
    #    print 'non-flag arguments:', argv

    chr_col = 0
    wb_gene_id_col = 1
    gene_name_col = 2
    strand_col = 3    
    gene_start_col = 4
    gene_end_col = 5
    isoform_tss_share_col = 6
    dcc_mut_tss_col = 8
    wt_emb_tss_col = 9
    wt_l1_tss_col = 10
    wt_l3_tss_col = 11
    dcc_mut_tss_diff_col = 13
    wt_emb_tss_diff_col =14
    wt_l1_tss_diff_col=15
    wt_l3_tss_diff_col=16
    allen_sl1_col=18
    allen_sl1_reads_col=19
    allen_sl1_sl2_variants_col=20
    dcc_mut_outron_len_col = 22
    wt_emb_outron_len_col = 23
    wt_l1_outron_len_col = 24
    wt_l3_outron_len_col =25

    translate_chr = {
    'CHROMOSOME_I':'chrI',
    'CHROMOSOME_II':'chrII',
    'CHROMOSOME_III': 'chrIII',
    'CHROMOSOME_IV': 'chrIV',
    'CHROMOSOME_V': 'chrV',
    'CHROMOSOME_X': 'chrX',
    'CHROMOSOME_MtDNA': 'mtDNA'}


    
    #Load worm genome from fasta file
    
    worm_genome = list(SeqIO.parse(FLAGS.genome_fasta,"fasta"))
    chr_dict = {}
    #for rec in worm_genome:
    #    print rec.id
    #    chr_dict[rec.id] = rec.seq 

    
        



    
    

    #Gather chromosome ids and tss coordinates
    wt_emb_tss_list = []
    with open (FLAGS.kruesi_file,'rb') as coords_file:
        csv_reader = csv.reader(coords_file,delimiter=',',quotechar='"')
        next(csv_reader) #Discard header line
        for i,row in enumerate(csv_reader):
            #if i==0:
            #     print row
            if row[wt_emb_tss_col] == '':
                tss_coord= -1
            else:
                tss_coord = int(row[wt_emb_tss_col])
            
            wt_emb_tss_list.append( [row[chr_col],
                                    row[gene_name_col],
                                    tss_coord] )


    for tss in wt_emb_tss_list:
        for rec in worm_genome:
            if (tss[0] == translate_chr[rec.id]):
                left_bound = tss[2]-300
                right_bound = tss[2]+300
                if (left_bound >0) and (tss[2] != -1):
                    print '\n>'+tss[1]+' 600 bp tss flanking region'
                    print str(rec.seq[left_bound:right_bound]).upper()

    
if __name__=='__main__':
    main(sys.argv)
