from dubioml.DuBioTools import DuBioTools

#Generate 3000 random sequences

n_examples = 6000

for i in range(n_examples):
    print ('>seq'+str(i)+'_neg')
    print (DuBioTools.rand_dna_nuc(600))
