import DuBioTools
from Bio import motifs
from Bio.Seq import Seq
import random

#Generate random sequences seeded with motifs at fixed positions relative to
#the center of the sequence.

def main():
    n_examples = 10000

    #For this data set, I will seed with 3 made-up motifs:
    motif1 = motifs.create([Seq('ATGATCG'),
                            Seq('ATGATCG'),
                            Seq('ATGATCG'),
                            Seq('TTGATCG'),
                            Seq('TAGATCG'),
                            Seq('TAGATCC')])

    motif2 = motifs.create([Seq('ACTGATAGA'),
                            Seq('AATGATAGA'),
                            Seq('ACTGATAGG'),
                            Seq('ACTGATAGC'),
                            Seq('ACTGATAGA'),
                            Seq('AGTGATAGA')])

    motif3 = motifs.create([Seq('CCGGCCGG'),
                            Seq('CCGGCCGG'),
                            Seq('CGGGCCAG'),
                            Seq('CCGGCCGA'),
                            Seq('ACGGCCGG')])

    motif4 = motifs.create([Seq('CGCCG'),
                            Seq('CGCCA')])

    motif1_prob = .15
    motif2_prob = .8
    motif3_prob = .25
    motif4_prob = .33

    for i in range(n_examples):
        #Generate a random 1000 bp long nucleotide sequence
        rand_seq = DuBioTools.rand_dna_nuc(600)
        
        #Maybe seed random sequence with each motif
        #Remember parameters are, nuc_seq, Bio.motif object, list of dist_from_tss,
        #orientation, and wobble
        if(decision(motif1_prob)):
            rand_seq = DuBioTools.rand_seed_motif(rand_seq,DuBioTools.rand_motif_instance(motif1),[-99],0,30)
        if(decision(motif2_prob)):
            rand_seq = DuBioTools.rand_seed_motif(rand_seq,
                           DuBioTools.rand_motif_instance(motif2),
                                                   [-184,210],0,12)
        if(decision(motif3_prob)):
            rand_seq = DuBioTools.rand_seed_motif(rand_seq,DuBioTools.rand_motif_instance(motif3),[-50],0,6)

        if(decision(motif4_prob)):
            rand_seq = DuBioTools.rand_seed_motif(rand_seq,DuBioTools.rand_motif_instance(motif4),[60],-1,7)
        print ('>seq'+str(i))
        print rand_seq

    return None#End main()
    
def decision(probability):
    return (random.random()<=probability)



if __name__ == '__main__':
    main()
