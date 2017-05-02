import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__),os.path.pardir))
from nucdata import *



def main():
    bed_file = "mmu-let-7b_clustered_chimeras_p13_cortex.bed"
    genome_file = "../../../Genomes/mm9/mm9.fa"
    chrom_sizes_file = "../../../Genomes/mm9/mm9.chrom.sizes"
    seq_len = 100
    
    
    nb = NucDataBedMem(bed_file,
                       genome_file,
                       chrom_sizes_file,
                       100,
                       skip_first=True,
                       gen_dinuc_shuffle=True)
    print nb.pull_index_nucs(5)

    #print nb.pull_index_onehot(5)

    print nb.pull_index_nucs(1234)

    print nb.pull_index_nucs(5123)
    


if __name__ == "__main__":
    main()
        
