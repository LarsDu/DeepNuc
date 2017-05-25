from Bio import SeqIO
import sys
import os

def main(argv):
    input_fasta = argv[1]
    output_fasta = os.path.splitext(input_fasta)[0]+"_fixed.fa"
    SeqIO.convert(input_fasta, "fasta", output_fasta, "fasta")


if __name__ == "__main__":
    main(sys.argv)
