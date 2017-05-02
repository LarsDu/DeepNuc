"""
This file contains methods for encoding nucleotides into the nibble format
for data compression and faster loading, particularly for HDF5 files 

Written by Lawrence Du



Note: The efficiency of this code may be greatly improved by replacing all
string and BitArray operations with lower level bit operators. A
tribit or 2bit encoding would also lead to even better compression
"""


import numpy as np
from bitstring import BitArray
import dubiotools as dbt

nib_dict = { 'T':'0000',
                'C':'0001',
                'A':'0010',
                'G':'0011',
                'N':'0100',
                't':'0000',
                'c':'0001',
                'a':'0010',
                'g':'0011',
                'n':'0100'
            }

nib_dict_rev = {0:'T',1:'C',2:'A',3:'G',4:'N'}
        
nibnuc_onehot_dict = { 0: np.array([1, 0, 0, 0]),
                       1: np.array([0, 1, 0, 0]),
                       2: np.array([0, 0, 1, 0]),
                       3: np.array([0, 0, 0, 1]),
                       4: np.array([.25, .25, .25, .25])
                      }


def main():
    seq = "ACTACGTTNTTTTTCTN"
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

    np_onehot = uint8_numpy_to_onehot_nucs(np_seq)
    print "Decoded numpy onehot", np_onehot
    print "Decoded numpy onehot shape", np_onehot.shape
    print "Back to nucs",dbt.onehot_to_nuc(np_onehot)
    
    
def nuc_to_nibble(seq_str):
    """Converts a nucletide string with alphabet ('T,C,A,G,N') to
    a binary string

    :param seq_str: A nucleotide string (ex: 'ATCTACTAC') 
    :returns: A binary string (ex:'0b01010101..')
    :rtype: string
    """
    seq_len = len(seq_str)
    bin_seq = ''
    bin_seq += format(seq_len,'#034b')
    for l in seq_str:
        bin_seq += nib_dict[l]
    return bin_seq



def nibble_to_nuc(bin_str):
    """Converts a nibble encoded string to a nucleotide string

    :param bin_str: A nibble encoded binary string where the first 32 bits
                    describe sequence length (ex: "0b10011001000010000001000010...")
    :returns: A Nucleotide string (ex: 'ACTACGTTTTTT...'
    :rtype: string

    """
    
    #Convert a quadbit encoded sequence to a nucleotide sequence
    bitarr = BitArray(bin_str)
    #Extract sequence length from first 32 bits
    #seq_len = bitarr[:32].uint
    
    nuc_str = ''
    for quad in bitarr[32:].cut(4):
        nuc_str += nib_dict_rev[quad.uint]
    return nuc_str
    

def nuc_to_uint8_numpy(seq_str):
    """Converts a nucleotide sequence to a numpy array. The first four
       uint8 digits encode a 32bit describing sequence length. The
       subsequent nibbles encode dinucleotides.

    :param seq_str: A nucleotide sequence with alphabet ('TACGN')
    :returns: A one row numpy array. The first four uint8 digits encode a 32bit
              describing sequence length. The subsequent nibbles  encode
              dinucleotides 
    :rtype: numpy array with dtype uint8

    """
    
    all_bits = BitArray(nuc_to_nibble(seq_str))
    seq_len = all_bits[:32].uint
    byte_list = []

    if seq_len%2 != 0 :
        #If odd length, pad out the last 4 bits.
        #These four bits will be truncated based on the encoded seq_len
        # in downstream operations
        all_bits += BitArray('0b0000')
    for byte in all_bits.cut(8):
        byte_list.append(byte.uint)
    return np.asarray(byte_list,dtype=np.uint8)

def byte_to_nucs(byte):
    """Convert a byte into a dinucleotide

    :param byte: A uint8
    :returns: A dinucleotide string
    :rtype: string
    """
    let1 = byte>>4
    let2 = byte &  15
    return nib_dict_rev[let1]+nib_dict_rev[let2]

def byte_to_onehot_nucs(byte):
    """Convert a byte into a dinucleotide

    :param byte: A uint8
    :returns: A dinucleotide string
    :rtype: string
    """
    let1 = byte>>4
    let2 = byte &  15
    r = np.zeros((4,2),dtype=np.float32)
    r[:,0] = nibnuc_onehot_dict[let1]
    r[:,1] = nibnuc_onehot_dict[let2]
    return r




def uint32_from_nums(byte_list):
    """Takes a list of four uint8 values and combines them into a
       32 bit value.

    :param byte_list: A list of four numbers 
    :returns: A 32 bit integer
    :rtype: int
    """
    
    if len(byte_list)!=4:
        print "byte_list must have four elements"
        return None
    bitarrs = ['0b']+[format(byte,'#010b')[2:] for byte in byte_list ]
    bit_string= ''.join(bitarrs)
    return int(bit_string,2)



def uint8_numpy_to_nucs(np_arr):
    """Convert a one row nibble encoded numpy array into a nucleotide sequence
    :param np_arr: One row numpy array. First 4 values encodes a 32-bit uint8 that
                   describes nucleotide sequence length
    :returns: A nucleotide string
    :rtype: string

    """
    
    byte_list = np_arr.tolist()
    #First four uint8s contain sequence length
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
                                      [byte_to_nucs(byte_list[-1])[0]]
        #Truncate final character
        return ''.join(nuc_list)

        
def uint8_numpy_to_onehot_nucs(np_arr):
    """
    Convert a one row nibble encoded numpy array of uint8s into a 4 x seq_len one-hot array
    Row order should be T,C,A,G

    :param np_arr: One row numpy array. First 4 values encodes a 32-bit uint8 that
                   describes nucleotide sequence length
    :returns: A nucleotide string
    :rtype: 4 x seq_len numpy array with dtype int32

    """
    
    byte_list = np_arr.tolist()
    #First four uint8s contain sequence length
    seq_len = uint32_from_nums(byte_list[:4])
    
    if seq_len%2 ==0:
        #If even length 
        nuc_list = [byte_to_onehot_nucs(int(byte)) for byte in byte_list[4:]]
        return np.concatenate(nuc_list,axis=1)
    else:
        #If odd seq_len, ignore last last 4 bits of final byte
        nuc_list = [byte_to_onehot_nucs(int(byte)) for byte in byte_list[4:-1]]+\
                                      [byte_to_onehot_nucs(byte_list[-1])]
        #Truncate final character
        return np.concatenate(nuc_list,axis=1)


    

if __name__ == "__main__":
    main()
