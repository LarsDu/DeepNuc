"""
Testing methods for converting
nucleotides >> nibble >> numpy array

T-000, C-001, A-010, G-011, N-111

This can help make HDF5 files w/ nucleotide information  slightly more compact

This is pretty inefficient code, but it should be faster than
saving onehot encoded numpy arrays!
"""


import numpy as np
from bitstring import BitArray

nib_dict = { 'T':'0000',
                'C':'0001',
                'A':'0010',
                'G':'0011',
                'N':'0100'}

nib_dict_rev = {0:'T',1:'C',2:'A',3:'G',4:'N'}
        


def main():
    seq = "ACTACGTTTTTTTTTTTTN"
    print seq
    print "Seq len",len(seq)
    bin_val =  nuc_to_nibble(seq)
    print bin_val
    rest_seq = nibble_to_nuc(bin_val)
    print rest_seq
    np_seq = nuc_to_uint8_numpy(seq)
    
    print "Numpy",np_seq
    print np_seq[0]

    np_decode = uint8_numpy_to_nucs(np_seq)
    print "Decoded numpy", np_decode
    print "Decoded numpy length", len(np_decode)
    
def nuc_to_nibble(seq_str):
    #first 32 bits describes length of sequence
    seq_len = len(seq_str)
    bin_seq = ''
    bin_seq += format(seq_len,'#034b')
    for l in seq_str:
        bin_seq += nib_dict[l]
    return bin_seq
    
def nibble_to_nuc(bin_str):
    #Convert a quadbit encoded sequence to a nucleotide sequence
    bitarr = BitArray(bin_str)
    #Extract sequence length from first 32 bits
    #seq_len = bitarr[:32].uint
    
    nuc_str = ''
    for quad in bitarr[32:].cut(4):
        nuc_str += nib_dict_rev[quad.uint]
    return nuc_str
    

def nuc_to_uint8_numpy(seq_str):
    all_bits = BitArray(nuc_to_nibble(seq_str))
    seq_len = all_bits[:32].uint
    byte_list = []

    if seq_len%2 != 0 :
        all_bits += BitArray('0b0000')
    for byte in all_bits.cut(8):
        byte_list.append(byte.uint)
    return np.asarray(byte_list,dtype=np.uint8)

def byte_to_nucs(byte):
    #byte is a uint8 (can also be an int)
    b = BitArray(format(byte,'#010b'))
    dinuc = ''
    for l in b.cut(4):
        dinuc += nib_dict_rev[l.uint]
    return dinuc

def leading_nib_to_nucs(byte):
    """Extracts first four bits of a byte and return nuc"""
    b = BitArray(format(byte,'#010b'))
    return nib_dict_rev[b[:4].uint]


def uint32_from_nums(byte_list):
    if len(byte_list)!=4:
        print "byte_list must have four elements"
        return None
    print byte_list
    bitarrs = ['0b']+[bin(byte)[2:] for byte in byte_list ]
    bit_string= ''.join(bitarrs)
    return int(bit_string,2)

def uint8_numpy_to_nucs(np_arr):
    #You need to know the length of the sequence!
    byte_list = np_arr.tolist()
    #First four elements contain sequence length
    seq_len = uint32_from_nums(byte_list[:4])

    #nuc_list = [byte_to_nucs(int(byte)) for byte in byte_list[4:]]
    #return ''.join(nuc_list)
    
    if seq_len%2 ==0:
        #If even length 
        nuc_list = [byte_to_nucs(int(byte)) for byte in byte_list[4:]] 
                               
        return ''.join(nuc_list)
    else:
        #If odd seq_len, ignore last last 4 bits of final byte
        nuc_list = [byte_to_nucs(int(byte)) for byte in byte_list[4:-1]]+\
                                      [ leading_nib_to_nucs(byte_list[-1]) ]
        #Truncate final character
        
        return ''.join(nuc_list)
    


if __name__ == "__main__":
    main()
