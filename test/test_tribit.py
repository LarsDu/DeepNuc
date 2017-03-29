"""
Note: ABANDON THIS. Too confusing to handle trailing bits not worth the time

Testing methods for converting
nucleotide >> tribit >> word (16 bit) coding for each 5 nucs >> numpy array
Note: the last bit for every 16 bits is unused
"""

import numpy as np
from bitstring import BitArray


tri_dict = { 'T':'000',
                'C':'001',
                'A':'010',
                'G':'011',
                'N':'100'}

tri_dict_rev = {0:'T',1:'C',2:'A',3:'G',4:'N'}



def main():
    seq = "GGACGTAGCTGACGTAC" #17 letters = 4 words
    print seq
    
    bin_val =  nuc_to_binary(seq)
    print bin_val
    
    rest_seq = tribit_to_nuc(bin_val)
    print rest_seq
    
    np_seq = nuc_to_word_numpy(seq)
    print np_seq
    print np_seq[0]

    np_decode = word_numpy_to_nucs(np_seq)
    print "Decoded numpy",np_decode
    

def nuc_to_binary(seq_str):
    bin_seq = '0b'
    spacer = (len(seq_str)*3)//15
    for i,l in enumerate(seq_str):
        bin_seq += tri_dict[l]
        if i%5 ==0:
            #Add spacer bit to make every 5 nucs =  16 bits
            bin_seq += '0'
        #Add end spacer
            
    return bin_seq

def words_to_nuc(bin_str):
    bitarr = BitArray(bin_str)
    nuc_str = ''
    for hexa in bitarr.cut(16):
        for tri in hexa[:-1].cut(3)  
            nuc_str += tri_dict_rev[tri.uint]
    return nuc_str

def nuc_to_word_numpy(seq_str):
    all_bits = nuc_to_tribit(seq_str)
    hex_list = []
    for pentadec in all_bits.cut(15):
        hex_list.append(pentadec.uint)
    return np.asarray(hex_list,dtype=word)

def word_to_nucs(word):
    b = BitArray(format(word,'#18b'))
    penta_nuc = ''
    #Discard last bit and cut every 3 bits
    for l in b[:-1].cut(3):
        penta_nuc += tri_dict_rev[l.uint]
    return penta_nuc

        
def word_numpy_to_nucs(np_seq):
    word_list = np_arr.tolist()
    nuc_list = [word_to_nucs(int(word,2)) for word in word_list]
    return ''.join(word_list)

if __name__ == "__main__":
    main()
