import numpy as np
import tensorflow as tf
"""
Note: remember to copy input nuc sequences before revcom ops
"""



def revcom_atgc(nuc):
	#nuc should be a b x 4 x n tensor 
	# with rows in the order A,T,G,C	
	
	rev_nuc = nuc_rev(nuc)
	return nuc_complement_atgc(rev_nuc)
	
def revcom_acgt(nuc):
	#nuc should be a b x 4 x n tensor 
	# with rows in the order A,C,G,T 
	
	rev_dims = tf.Constant([False,True,True])
	return tf.reverse(nuc,rev_dims)
	
def nuc_rev(nuc):
	#nuc should be a b x 4 x n tensor 
	
	#Reverse nuc sequence
	rev_dims = tf.Constant([False,False,True],dtype=tf.bool)
	return tf.reverse(nuc,rev_dims)
	
	
def nuc_complement_atgc(nuc):					
	# nuc should be a one-hot b x 4 x n tensor 	
	# with rows in the order A,T,G,C
	# Note: this is very inefficient compared to coding
	# nuc matrices as ACGT or AGCT
	a_ten,t_ten,g_ten,c_ten = tf.unstack(nuc, None, axis=1)
	nuc_com = tf.pack([t_ten,a_ten,c_ten,g_ten],axis=1)
	return nuc_com
	
def nuc_complement_acgt(nuc):					
	# nuc should be a b x 4 x n tensor 
	# with rows in the order A,C,G,T 
	# Note: should be faster than nuc_complement atgc
	rev_dims = tf.Constant([False,True,False])
	return tf.reverse(nuc,rev_dims)
	
