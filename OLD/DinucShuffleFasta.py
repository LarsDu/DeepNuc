import DinucShuffle as dinuc
import sys

from Bio import SeqIO

fname = str(sys.argv[1])
print "Input file is ", fname 
seq_parser = SeqIO.parse(fname,"fasta")
for record in seq_parser:
    #Writes dinucleotide shuffled version of each input sequence
    #to stddout
    A= dinuc.generate_sequences(seq_parser,1)


seq_parser.close()
