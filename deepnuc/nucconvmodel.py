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
        nn = dtl.Network(x_nuc,[drop_out],bounds=[0.,1.])
        
        logits = nn.forward()
        return logits,nn



def inferenceB(dna_seq_placeholder,keep_prob_placeholder,num_classes):
    with tf.variable_scope("inference_b") as scope:
        print "Running inferenceB - strided convolutions with wide filters"

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
        nn = dtl.Network(x_nuc,[drop_out],bounds=[0.,1.])
        logits = nn.forward()
        return logits,nn




def inferenceC(dna_seq_placeholder,keep_prob_placeholder,num_classes):
    """2-strided convolutions with small filters (size 3) """

    with tf.variable_scope("inference_c") as scope:
        print "Utilizing inferenceC_600bp"


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
        nn = dtl.Network(x_nuc,[drop_out],bounds=[0.,1.])
        logits = nn.forward()

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
        nn = dtl.Network(x_nuc,[dropout],bounds=[0.,1.])
        logits=nn.forward()
        return logits,nn





#methods_dict is used by modelparams.py
methods_dict = {"inferenceA":inferenceA,
               "inferenceB":inferenceB,
               "inferenceC":inferenceC,
               "inferenceD":inferenceD}



'''
def loss(logits, labels):
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels,
    #                                                        name='cross_entropy')


    #cross_entropy = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(tf.nn.softmax(logits),1e-10,1.0)))
    #To use scalar summary, first argument needs to be a list
    #with same shape as cross_entropy
    #tf.scalar_summary(cross_entropy.op.name, cross_entropy)
    #cross_entropy = -tf.reduce_sum(labels * tf.log(logits), reduction_indices=[1])
    loss = tf.reduce_mean(cross_entropy,
                          name='xentropy_mean')
    dtl.activation_summary(loss)
    return loss

def training(loss,learning_rate):
    #Create a scalar summary for loss function
    #tf.scalar_summary(loss.op.name, loss)
    tf.summary.scalar(loss.op.name,loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss,global_step = global_step)
    return train_op

def training_adam(loss,learning_rate):
    #Create a scalar summary for loss function
    #tf.scalar_summary(loss.op.name, loss)
    tf.summary.scalar(loss.op.name,loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss,global_step = global_step)
    return train_op


def logits_to_probs(logits):
    return tf.sigmoid(logits)

def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
    range [0, NUM_CLASSES).
    Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    #correct = tf.nn.in_top_k(logits, tf.cast(labels,tf.int32), 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))
'''
