import numpy as np
import tensorflow as tf
import dtlayers as dtl 



def inferenceA(dna_seq_placeholder,
               keep_prob_placeholder,
               num_classes,
               concat_revcom = False):
    with tf.variable_scope("inference_a",reuse=None) as scope:
        
        print "Running inferenceA"
        dna_conv_filter_width = 24
        num_dna_filters = 48
        pad_size=0

        #Reshape and pad nucleotide input
        x_nuc = dtl.NucInput(dna_seq_placeholder,pad_size,'dna_input',concat_revcom)
        cl1 = dtl.Conv2d(x_nuc,
                       filter_shape=[1,
                                     dna_conv_filter_width,
                                     4,
                                     num_dna_filters],
                       strides = [1,1,1,1],
                       padding = 'SAME',
                       name='dna_conv1')
        r1 = dtl.Relu(cl1)
        p1 = dtl.AvgPool(r1,[1,4],'dna_avg_pool1')
        cl2 = dtl.Conv2d(p1,
                       filter_shape = [1,
                                       dna_conv_filter_width,
                                       num_dna_filters,
                                       num_dna_filters],
                      strides = [1,1,1,1],
                      padding='SAME',
                       name='dna_conv2')
        r2 = dtl.Relu(cl2)
        p2 = dtl.AvgPool(r2,[1,4],'dna_avg_pool2')
        flat = dtl.Flatten(p2)
        l1 = dtl.Linear(flat,100,'linear1')
        r3 = dtl.Relu(l1)
        l2 = dtl.Linear(r3,50,'linear2' )  
        r4 = dtl.Relu(l2)
        l3 = dtl.Linear(r4,num_classes,'linear3' )
        drop_out = dtl.Dropout(l3,keep_prob_placeholder,name="dropout")
        #nn = dtl.Network(x_nuc,[cl1,r1,p1, cl2,r2,p2,flat,l1,r3,l2,r4,l3],bounds=[0.,1.])
        nn = dtl.Network(x_nuc,[drop_out],bounds=[0.,1.],use_zbeta_first=False)
        
        logits = nn.forward()
        return logits,nn


def inferenceB(dna_seq_placeholder,
               keep_prob_placeholder,
               num_classes,
               concat_revcom = False):

    with tf.variable_scope("inference_b",reuse=None) as scope:
        print "Running inference B"
        print "Simple 2 layer linear network with no dropout"
        x_nuc = dtl.NucInput(dna_seq_placeholder,pad_size,'dna_input',concat_revcom)
        flat = dtl.Flatten(x_nuc)
        l1 = dtl.Linear(flat,1024,'linear1')
        r1 = dtl.Relu(l1)
        l2 = dtl.Linear(r1,1024,'linear2')
        r2 = dtl.Relu(l2)
        nn = dtl.Network(x_nuc,[r2],bounds=[0.,1.],use_zbeta_first=False)
        logits = nn.forward()
        return logits,nn 
        
        

def inferenceC(dna_seq_placeholder,
               keep_prob_placeholder,
               num_classes,
               concat_revcom = False):

    with tf.variable_scope("inferenceC",reuse=None) as scope:
        print "Running inference C"
        x_nuc = dtl.NucInput(dna_seq_placeholder,pad_size,'dna_input',concat_revcom)

        dna_conv_filter_width = 12
        num_dna_filters = 72
        pad_size=0

        cl1 = dtl.Conv2d(x_nuc,
               filter_shape=[1,
                             dna_conv_filter_width,
                             4,
                             num_dna_filters],
               strides = [1,1,1,1],
               padding = 'SAME',
               name='dna_conv1')
        
        r1 = dtl.Relu(cl1)
        p1 = dtl.AvgPool(r1,[1,4],'dna_avg_pool1')

        flat = dtl.Flatten(p1)
        l1 = dtl.Linear(flat,512,'linear1')
        r2 = dtl.Relu(l1)
        l2 = dtl.Linear(r2,512,'linear2')
        r3 = dtl.Relu(l2)
        l3 = dtl.Linear(r3,512,'linear3')
        r4 = dtl.Relu(l3)
        readout = dtl.Linear(r4,num_classes,'readout')
        dropout = dtl.Dropout(readout,keep_prob_placeholder,name="dropout")
        nn = dtl.Network(x_nuc,[dropout],bounds=[0.,1.],use_zbeta_first=False)
        logits=nn.forward()
        return logits,nn
            

def inferenceD(dna_seq_placeholder,
               keep_prob_placeholder,
               num_classes,
               concat_revcom = False):
    with tf.variable_scope("inference_d",reuse=None) as scope:
        
        print "Running inferenceD"
        dna_conv_filter_width = 20
        num_dna_filters = 16
        pad_size=0

        x_nuc = dtl.NucInput(dna_seq_placeholder,pad_size,'dna_input',concat_revcom)
        cl1 = dtl.Conv2d(x_nuc,
               filter_shape=[1,
                             dna_conv_filter_width,
                             4,
                             num_dna_filters],
               strides = [1,1,1,1],
               padding = 'SAME',
               name='dna_conv1')
        
        r1 = dtl.Relu(cl1)
        p1 = dtl.AvgPool(r1,[1,4],'dna_avg_pool1')

        flat = dtl.Flatten(p1)
        l1 = dtl.Linear(flat,32,'linear1')
        r2 = dtl.Relu(l1)
        readout = dtl.Linear(r2,num_classes,'readout')
        dropout = dtl.Dropout(readout,keep_prob_placeholder,name="dropout")
        nn = dtl.Network(x_nuc,[dropout],bounds=[0.,1.],use_zbeta_first=False)
        logits=nn.forward()
        return logits,nn






def inferenceE(dna_seq_placeholder,keep_prob_placeholder,num_classes):
    with tf.variable_scope("inference_e") as scope:
        print "Running inferenceE - strided convolutions with wide filters"

        dna_conv_filter_width = 96
        num_dna_filters = 192
        pad_size=0

        #Reshape and pad nucleotide input
        x_nuc = dtl.NucInput(dna_seq_placeholder,pad_size,'dna_input')
        cl1 = dtl.Conv2d(x_nuc,
                       filter_shape=[1,
                                     dna_conv_filter_width,
                                     4,
                                     num_dna_filters],
                       strides = [1,1,1,1],
                       padding = 'SAME',
                       name='dna_conv1')
        r1 = dtl.Relu(cl1)
        p1 = dtl.AvgPool(r1,[1,4],'dna_avg_pool1')
        cl2 = dtl.Conv2d(p1,
                       filter_shape = [1,
                                       dna_conv_filter_width,
                                       num_dna_filters,
                                       num_dna_filters],
                      strides = [1,1,1,1],
                      padding='SAME',
                       name='dna_conv2')
        r2 = dtl.Relu(cl2)
        p2 = dtl.AvgPool(r2,[1,4],'dna_avg_pool2')
        flat = dtl.Flatten(p2)
        l1 = dtl.Linear(flat,100,'linear1')
        r3 = dtl.Relu(l1)
        l2 = dtl.Linear(r3,50,'linear2' )  
        r4 = dtl.Relu(l2)
        l3 = dtl.Linear(r4,num_classes,'linear3' )
        drop_out = dtl.Dropout(l3,keep_prob_placeholder,name="dropout")
        #nn = dtl.Network(x_nuc,[cl1,r1,p1, cl2,r2,p2,flat,l1,r3,l2,r4,l3],bounds=[0.,1.])
        nn = dtl.Network(x_nuc,[drop_out],bounds=[0.,1.],use_zbeta_first=False)
        logits = nn.forward()
        return logits,nn



def inferenceF(dna_seq_placeholder,keep_prob_placeholder,num_classes):
    """2-strided convolutions with small filters (size 3) """

    with tf.variable_scope("inference_f") as scope:
        print "Utilizing inferencef_600bp"


        pad_size = 0
        seq_len = dna_seq_placeholder.get_shape().as_list()[2]
        if seq_len != 600:
            print "Sequence length does not equal 600"
            print "Possible errors"

        nf1 = 24
        nf2 = 48
        stride = 2
        fw1=3
        fw2=6
        num_classes =2

        x_nuc = dtl.NucInput(dna_seq_placeholder,pad_size,'dna_input')

        cl1 = dtl.Conv2d(x_nuc,filter_shape = [1,fw1,4,nf1],
                             strides = [1,1,1,1],
                             padding = 'SAME',
                             name = 'dna_conv1')
        r1 = dtl.Relu(cl1)


        cl2 = dtl.Conv2d(r1,filter_shape = [1,fw1,nf1,nf1],
                                     strides = [1,1,stride,1],
                                        padding = 'SAME',
                                        name = 'dna_conv2')
        r2 = dtl.Relu(cl2)


        cl3 = dtl.Conv2d(r2,filter_shape= [1,fw1,nf1,nf1],
                          strides = [1,1,stride,1],
                          padding = 'SAME',
                          name = 'dna_conv3') 
        r3 = dtl.Relu(cl3)

        cl4 = dtl.Conv2d(r3,filter_shape=[1,fw1,nf1,nf2],
                       strides = [1,1,1,1],
                          padding = 'SAME',
                          name = 'dna_conv4')
        r4 = dtl.Relu(cl4)

        cl5 = dtl.Conv2d(r4,filter_shape=[1,fw1,nf2,nf2],
                            strides = [1,1,stride,1],
                            padding = 'SAME',
                            name = 'dna_conv5')
        r5 = dtl.Relu(cl5)

        cl6 = dtl.Conv2d(r5,filter_shape=[1,fw1,nf2,nf2],
                         strides = [1,1,stride,1],
                         padding = 'SAME',
                         name = 'dna_conv6')
        r6 = dtl.Relu(cl6)


        flat = dtl.Flatten(r6)

        l2 = dtl.Linear(flat,nf2,'linear2' )  
        r4 = dtl.Relu(l2)
        l3 = dtl.Linear(r4,num_classes,'linear3' )

        drop_out = dtl.Dropout(l3,keep_prob_placeholder,name="dropout")

        #Passing the last layer to the Network constructor
        #will automatically initialize Network with every preceding layer
        nn = dtl.Network(x_nuc,[drop_out],bounds=[0.,1.],use_zbeta_first=False)
        logits = nn.forward()

        return logits,nn



#methods_dict is used by modelparams.py
methods_dict = {
               "inferenceA":inferenceA,
               "inferenceB":inferenceB,
               "inferenceC":inferenceC,
               "inferenceD":inferenceD,
               "inferenceE":inferenceE,
               "inferenceF":inferenceF,
               }



