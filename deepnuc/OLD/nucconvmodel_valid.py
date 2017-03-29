import numpy as np
import tensorflow as tf
import dtlayers as dtl 




def inferenceA(dna_seq_placeholder,keep_prob_placeholder,num_classes):
    print "Running inferenceA"

    dna_conv_filter_width = 24
    num_dna_filters = 48
    pad_size=0
    
    #Reshape and pad nucleotide input
    x_nuc = dtl.NucInput(dna_seq_placeholder,pad_size,'dna_input')
    cl1 = dtl.Conv(x_nuc,
                   filter_shape=[1,
                                 dna_conv_filter_width,
                                 4,
                                 num_dna_filters],
                   strides = [1,1,1,1],
                   padding = 'VALID',
                   name='dna_conv1')
    r1 = dtl.Relu(cl1)
    p1 = dtl.AvgPool(r1,[1,4],'dna_avg_pool1')
    cl2 = dtl.Conv(p1,
                   filter_shape = [1,
                                   dna_conv_filter_width,
                                   num_dna_filters,
                                   num_dna_filters],
                  strides = [1,1,1,1],
                  padding='VALID',
                   name='dna_conv2')
    r2 = dtl.Relu(cl2)
    p2 = dtl.AvgPool(r2,[1,4],'dna_avg_pool2')
    flat = dtl.Flatten(p2)
    l1 = dtl.Linear(flat,100,'linear1')
    r3 = dtl.Relu(l1)
    l2 = dtl.Linear(r3,50,'linear2' )  
    r4 = dtl.Relu(l2)
    l3 = dtl.Linear(r4,num_classes,'linear3' )

    #nn = dtl.Network(x_nuc,[cl1,r1,p1, cl2,r2,p2,flat,l1,r3,l2,r4,l3],bounds=[0.,1.])
    nn = dtl.Network(x_nuc,[l3],bounds=[0.,1.])
    logits = nn.forward()
    return logits,nn


def inferenceB_600bp(dna_seq_placeholder,keep_prob_placeholder,num_classes):
    print "Utilizing inferenceB_600bp"
    """A fully convolutional network for DNA"""

    pad_size=12
    x_nuc = dtl.NucInput(dna_seq_placeholder,pad_size,'dna_input')
    #x_nuc shape should be 1,1,624,4
    cl1 = dtl.Conv(x_nuc,filter_shape = [1,20,4,40],                       
                         padding = 'VALID',
                         name = 'dna_conv1')
    r1 = dtl.Relu(cl1)
    p1 = dtl.AvgPool(r1,[1,4],'dna_avg_pool1')

    cl2 = dtl.Conv(p1,filter_shape=[1,15,40,100],
                                    padding = 'VALID',
                                    name='dna_conv2')
    r2 = dtl.Relu(cl2)
    p2 = dtl.AvgPool(r2,[1,4],'dna_avg_pool2')

    cl3 = dtl.Conv(p2,filter_shape= [1,8,100,48],
                      padding = 'VALID',
                      name='dna_conv3') 
    r3 = dtl.Relu(cl3)
    p3 = dtl.AvgPool(r3,[1,4],'dna_avg_pool3')

    cl4 = dtl.Conv(p3,filter_shape=[1,6,48,1],
                      padding = 'VALID',
                      name = 'dna_conv4')

    flat = dtl.Flatten(cl4)

    #Passing the last layer to the Network constructor
    #will automatically initialize Network with every preceding layer
    nn = dtl.Network(x_nuc,[flat],bounds=[0.,1.])
    logits = nn.forward()

    return logits,nn


def inferenceC_600bp(dna_seq_placeholder,keep_prob_placeholder,num_classes):
    print "Running inferenceC for 600 bp input sequence"
    pad_size=12
    x_nuc = dtl.NucInput(dna_seq_placeholder,pad_size,'dna_input')

    #Block 1
    c1 = dtl.Conv(x_nuc,
                  filter_shape = [1,30,4,32],
                  strides = [1,1,1,1],
                  padding = 'VALID',
                  name = 'dna_conv1')
    r1 = dtl.Relu(c1)
    c2 = dtl.Conv(r1,
                  filter_shape=[1,30,32,32],
                  strides = [1,1,1,1],
                  padding = 'VALID',
                  name = 'dna_conv2')
    r2 = dtl.Relu(c2)
    c3s = dtl.Conv(r2,
                  filter_shape = [1,30,32,32],
                  strides = [1,1,4,1],
                  padding = 'VALID',
                  name = 'dna_conv3_strided')
    #Replace pooling op with striding
    r3 = dtl.Relu(c3s)

    #Block 2
    c4 = dtl.Conv(r3,
                  filter_shape= [1,25,32,48],
                  strides = [1,1,1,1],
                  padding = 'VALID',
                  name = 'dna_conv4')
    r4 = dtl.Relu(c4)
    c5 = dtl.Conv(r4,
                  filter_shape=[1,25,48,48],
                  strides = [1,1,1,1],
                  padding= 'VALID',
                  name = 'dna_conv5')
    r5 = dtl.Relu(c5)
    c6s = dtl.Conv(r5,
                   filter_shape = [1,25,48,48],
                   strides = [1,1,4,1],
                   padding = 'VALID',
                   name = 'dna_conv6_strided')
    r6 = dtl.Relu(c6s)

    #Block 3
    c7 = dtl.Conv(r6,
                  filter_shape=[1,16,48,24],
                  strides = [1,1,1,1],
                  padding = 'VALID',
                  name = 'dna_conv7')
    r7 = dtl.Relu(c7)
    c8 = dtl.Conv(r7,
                  filter_shape =[1,1,24,2],
                  strides = [1,1,1,1],
                  name = 'dna_conv8')
    r8 = dtl.Relu(c8)
    drop = dtl.Dropout(r8,keep_prob_placeholder)
    flat = dtl.Flatten(drop)
    nn = dtl.Network(x_nuc,[flat],bounds=[0.,1.])
    logits = nn.forward()
    return logits,nn



def inferenceC_600bp_tester(dna_seq_placeholder,keep_prob_placeholder,num_classes):
    print "Running inferenceC for 600 bp input sequence"
    pad_size=0
    x_nuc = dtl.NucInput(dna_seq_placeholder,pad_size,'dna_input')

    nf1=24
    nf2=48

    #Block 1
    c1 = dtl.Conv(x_nuc,
                  filter_shape = [1,13,4,nf1],
                  strides = [1,1,1,1],
                  padding = 'VALID',
                  name = 'dna_conv1')
    r1 = dtl.Relu(c1)
    c2 = dtl.Conv(r1,
                  filter_shape=[1,13,nf1,nf1],
                  strides = [1,1,1,1],
                  padding = 'VALID',
                  name = 'dna_conv2')
    r2 = dtl.Relu(c2)
    c3s = dtl.Conv(r2,
                  filter_shape = [1,13,nf1,nf1],
                  strides = [1,1,7,1],
                  padding = 'VALID',
                  name = 'dna_conv3_strided')
    #Replace pooling op with striding
    r3 = dtl.Relu(c3s)

    #Block 2
    c4 = dtl.Conv(r3,
                  filter_shape= [1,7,nf1,nf2],
                  strides = [1,1,1,1],
                  padding = 'VALID',
                  name = 'dna_conv4')
    r4 = dtl.Relu(c4)
    c5 = dtl.Conv(r4,
                  filter_shape=[1,7,nf2,nf2],
                  strides = [1,1,1,1],
                  padding= 'VALID',
                  name = 'dna_conv5')
    r5 = dtl.Relu(c5)
    c6s = dtl.Conv(r5,
                   filter_shape = [1,7,nf2,nf2],
                   strides = [1,1,5,1],
                   padding = 'VALID',
                   name = 'dna_conv6_strided')
    r6 = dtl.Relu(c6s)

    #Block 3
    c7 = dtl.Conv(r6,
                  filter_shape=[1,5,nf2,nf2],
                  strides = [1,1,3,1],
                  padding = 'VALID',
                  name = 'dna_conv7_strided')
    r7 = dtl.Relu(c7)


    c7b = dtl.Conv(r7,
                   filter_shape=[1,3,nf2,nf2],
                   strides=[1,1,2,1],
                   padding='VALID',
                   name='dna_conv7b_strided')

    r7b = dtl.Relu(c7b)

    #Engineer this section to reduce sequence to 1x1
    c8 = dtl.Conv(r7b,
                  filter_shape =[1,1,nf2,nf2],
                  strides = [1,1,1,1],
                  name = 'dna_conv8')

    r8 = dtl.Relu(c8)
    c9 = dtl.Conv(r8,
                  filter_shape =[1,1,nf2,2],
                  strides = [1,1,1,1],
                  name = 'dna_conv9_final')

    r8 = dtl.Relu(c9)


    #drop = dtl.Dropout(r8,keep_prob_placeholder)
    #flat = dtl.Flatten(drop)
    flat = dtl.Flatten(r8)
    nn = dtl.Network(x_nuc,[flat],bounds=[0.,1.])
    logits = nn.forward()
    return logits,nn



def inferenceC_600bp_lite(dna_seq_placeholder,keep_prob_placeholder):
    print "Running inferenceC for 600 bp input sequence"
    pad_size=12
    x_nuc = dtl.NucInput(dna_seq_placeholder,pad_size,'dna_input')

    #Block 1
    c1 = dtl.Conv(x_nuc,
                  filter_shape = [1,30,4,24],
                  strides = [1,1,1,1],
                  padding = 'VALID',
                  name = 'dna_conv1')
    r1 = dtl.Relu(c1)
    c2 = dtl.Conv(r1,
                  filter_shape=[1,25,24,24],
                  strides = [1,1,1,1],
                  padding = 'VALID',
                  name = 'dna_conv2')
    r2 = dtl.Relu(c2)
    c3s = dtl.Conv(r2,
                  filter_shape = [1,30,24,24],
                  strides = [1,1,4,1],
                  padding = 'VALID',
                  name = 'dna_conv3_strided')
    #Replace pooling op with striding
    r3 = dtl.Relu(c3s)

    #Block 2
    c4 = dtl.Conv(r3,
                  filter_shape= [1,25,24,36],
                  strides = [1,1,1,1],
                  padding = 'VALID',
                  name = 'dna_conv4')
    r4 = dtl.Relu(c4)
    c5 = dtl.Conv(r4,
                  filter_shape=[1,25,36,36],
                  strides = [1,1,1,1],
                  padding= 'VALID',
                  name = 'dna_conv5')
    r5 = dtl.Relu(c5)
    c6s = dtl.Conv(r5,
                   filter_shape = [1,25,36,24],
                   strides = [1,1,4,1],
                   padding = 'VALID',
                   name = 'dna_conv6_strided')
    r6 = dtl.Relu(c6s)

    #Block 3
    c7 = dtl.Conv(r6,
                  filter_shape=[1,16,24,24],
                  strides = [1,1,1,1],
                  padding = 'VALID',
                  name = 'dna_conv7')
    r7 = dtl.Relu(c7)
    c8 = dtl.Conv(r7,
                  filter_shape =[1,1,24,2],
                  strides = [1,1,1,1],
                  name = 'dna_conv8')
    r8 = dtl.Relu(c8)
    drop = dtl.Dropout(r8,keep_prob_placeholder)
    flat = dtl.Flatten(r8)
    nn = dtl.Network(x_nuc,[flat],bounds=[0.,1.])
    logits = nn.forward()
    return logits,nn



def inferenceD(dna_seq_placeholder,keep_prob_placeholder,num_classes):

    """
    Attemp to improve inferenceA
    """


    fw1 = 24
    fw2 = 12
    nf1 = 192
    nf2=96
    pad_size=0
    x_nuc = dtl.NucInput(dna_seq_placeholder,pad_size,'dna_input')
    cl1 = dtl.Conv(x_nuc,
                   filter_shape=[1,
                                 fw1,
                                 4,
                                 nf1],
                   strides = [1,1,1,1],
                   padding = 'VALID',
                   name='dna_conv1')
    r1 = dtl.Relu(cl1)
    p1 = dtl.AvgPool(r1,[1,4],'dna_avg_pool1')
    cl2 = dtl.Conv(p1,
                   filter_shape = [1,
                                   fw2,
                                   nf1,
                                   nf1],
                  strides = [1,1,1,1],
                  padding='VALID',
                   name='dna_conv2')
    r2 = dtl.Relu(cl2)

    p2 = dtl.AvgPool(r2,[1,4],'dna_avg_pool2')
    flat = dtl.Flatten(p2)
    l1 = dtl.Linear(flat,nf2,'linear1')
    r3 = dtl.Relu(l1)
    l2 = dtl.Linear(r3,nf2,'linear2' )  
    r4 = dtl.Relu(l2)
    l3 = dtl.Linear(r4,num_classes,'linear3' )
    drop = dtl.Dropout(l3,keep_prob_placeholder,'dropout')
    nn = dtl.Network(x_nuc,[drop],bounds=[0.,1.])
    logits = nn.forward()
    return logits,nn



def inferenceE(dna_seq_placeholder,keep_prob_placeholder,num_classes):


    """
    Attempt to improve inferenceD by removing pooling after first conv layer,
    and adding a third conv layer after avg pooling second conv layer
    Note: This seems to work pretty decently
    """


    fw1 = 24
    fw2 = 12
    nf1 = 192
    nf2=96
    pad_size=0
    x_nuc = dtl.NucInput(dna_seq_placeholder,pad_size,'dna_input')
    cl1 = dtl.Conv(x_nuc,
                   filter_shape=[1,
                                 fw1,
                                 4,
                                 nf2],
                   strides = [1,1,1,1],
                   padding = 'VALID',
                   name='dna_conv1')
    r1 = dtl.Relu(cl1)

    cl2 = dtl.Conv(r1,
                   filter_shape = [1,
                                   fw1,
                                   nf2,
                                   nf2],
                  strides = [1,1,1,1],
                  padding='VALID',
                   name='dna_conv2')
    r2 = dtl.Relu(cl2)
    p1 = dtl.AvgPool(r2,[1,4],'dna_avg_pool1')

    cl3 = dtl.Conv(p1,
                   filter_shape = [1,
                                   fw2,
                                   nf2,
                                   nf1],
                  strides = [1,1,1,1],
                  padding='VALID',
                  name='dna_conv3')

    r3 = dtl.Relu(cl3)
    p2 = dtl.AvgPool(r3,[1,4],'dna_avg_pool2')
    flat = dtl.Flatten(p2)
    l1 = dtl.Linear(flat,nf1,'linear1')
    r3 = dtl.Relu(l1)
    l2 = dtl.Linear(r3,nf2,'linear2' )  
    r4 = dtl.Relu(l2)
    l3 = dtl.Linear(r4,num_classes,'linear3' )
    drop = dtl.Dropout(l3,keep_prob_placeholder,'dropout')
    nn = dtl.Network(x_nuc,[drop],bounds=[0.,1.])
    logits = nn.forward()
    return logits,nn





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
