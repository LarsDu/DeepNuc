import sys
import os.path
sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import tfbiotools
import dubiotools as dbt
import tensorflow as tf
import numpy as np

import dtlayers as dtl

def main():
    nucs = ['ATACGACGACT','GTACGACGTAC','GTCGATGCATG']
    nuc_3d = np.zeros(  ( len(nucs),len(nucs[0]),4 )   )
    for i,nuc in enumerate(nucs):
        nuc_3d[i,:,:] = dbt.seq_to_onehot_topdown(nuc)

    nuc_3d_ten = tf.stack(nuc_3d)

    #test_reverse_sequence(nuc_3d)

    print "Test complement"
    #test_com(nuc_3d_ten)

    print "Test reverse complement"
    #test_revcom(nuc_3d_ten)

    #test_nuc_input(nuc_3d_ten)


    print "Test seq to onehot"
    print dbt.seq_to_onehot('AACGTCG').T
    






def test_nuc_input(nucs):
    with tf.Session() as sess:
        
        l1 = dtl.NucInput(nucs,pad_size=0,name='nuc_input',concat_revcom=True)
        out  = sess.run(l1.output)
        print "Outshape",out.shape

    
    #print np.squeeze(out,axis=1).shape
    print "New seqs", dbt.onehot_4d_to_nuc(out)


    
def test_reverse_sequence(nuc_3d):
    with tf.Session() as sess:
        
        rev_seq = sess.run(tf.reverse(nuc_3d,[1]))
        print rev_seq

    print "Original"
    print nuc_3d 
    print nuc_3d.shape

    print "Reversed"
    print rev_seq
    print rev_seq.shape
    


def test_com(nuc_3d_ten):
    with tf.Session() as sess:
        nuc_3d = sess.run(nuc_3d_ten)
        complement_seq = sess.run(tfbiotools.complement_onehot_tcag(nuc_3d_ten))
    print_compare_3d(nuc_3d,complement_seq)
        

def test_revcom(nuc_3d_ten):
    with tf.Session() as sess:
        nuc_3d = sess.run(nuc_3d_ten)
        revcom_seq = sess.run(tfbiotools.revcom_onehot_tcag(nuc_3d_ten))
    
    print_compare_3d(nuc_3d,revcom_seq)

def print_compare_3d(np_orig,np_new):
    if np_orig.shape==np_new.shape:
        print "Shapes match",np_orig.shape
    else:
        print "Shape mismatch",np_orig.shape,np_new.shape
        

    print "Original sequences"
    print dbt.onehot_4d_to_nuc(np.expand_dims(np_orig,1))

    print "New sequences"
    print dbt.onehot_4d_to_nuc(np.expand_dims(np_new,1))

    
if __name__ == "__main__":
    main()
  
