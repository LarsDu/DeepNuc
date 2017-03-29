"""
Implementation of a Convolutional Neural Network with DeepTaylor decomposition 

Based off the tutorial
http://tensorflow.org/tutorials/mnist/beginners/index.md

Written by Lawrence Du 

"""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
#import argparse


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import dtlayers as dtl
import argparse

#For seeing z-values in figures on mouseover
from FormatPlot import Formatter 

from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string('data_dir',"/tmp/data","""MNIST data directory""")
flags.DEFINE_string('mode','train',"""Options: \'train\' or \'visualize\' """)
flags.DEFINE_string('save_dir','mnist_test2',"""Directory under which to place checkpoints""")

flags.DEFINE_integer('num_iterations',100000,""" Number of training iterations """)
flags.DEFINE_integer('num_visualize',10,""" Number of samples to visualize""")
flags.DEFINE_integer('batch_size',25,""" Number of samples to visualize""")


def main(_):
   
    mode = FLAGS.mode
    num_visualize = FLAGS.num_visualize
    num_iterations = FLAGS.num_iterations
    
    save_dir = FLAGS.save_dir
    checkpoint_dir = save_dir+"/checkpoints"
    summary_dir =save_dir+"/summaries"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    x_image = dtl.ImageInput(x,image_shape=[28,28,1])
    
  
    #conv filter dimensions are  w,h,input_dims,output_dims
    
    #Example replacing pooling layers with strided conv layers

    nf1=96   #NEED A BIGGER VID CARD
    nf2 = 192 # NEED A BIGGER VID CARD
    #nf1=48 #NEED A BIGGER VID CARD
    #nf2=24 #NEED A BIGGER VID CARD
    #nf1=32
    #nf2=24

    #Block1
    cl1 = dtl.Conv(x_image, filter_shape = [3,3,1,nf1],padding = 'VALID',name='conv1')
    r1 = dtl.Relu(cl1)
    cl2 = dtl.Conv(r1, filter_shape = [3,3,nf1,nf1],padding = 'VALID',name='conv2')
    r2 = dtl.Relu(cl2)
    cl3s = dtl.Conv(r2,filter_shape=[3,3,nf1,nf1],
                      strides = [1,2,2,1],
                      padding = 'VALID',
                      name = 'conv3_strided')
    r3 = dtl.Relu(cl3s)

    #Block2
    cl4 = dtl.Conv(r3, filter_shape = [3,3,nf1,nf2],
                       strides=[1,1,1,1],
                       padding='VALID',name='conv4')
    r4 = dtl.Relu(cl4)
    cl5 = dtl.Conv(r4, filter_shape = [3,3,nf2,nf2],
                       strides=[1,1,1,1],
                       padding='VALID',name='conv5')
   
    r5 = dtl.Relu(cl5)
    cl6s = dtl.Conv(r5, filter_shape = [3,3,nf2,nf2],
                       strides=[1,2,2,1],
                       padding='VALID',name='conv6_strided')

    r6 = dtl.Relu(cl6s)
    c7 = dtl.Conv(r6,filter_shape=[3,3,nf2,nf2],
                      strides = [1,2,2,1],
                      padding = 'VALID',
                      name = 'conv7_strided')
    r7 = dtl.Relu(c7)


    c8 = dtl.Conv(r7,filter_shape =[1,1,nf2,nf2],
                  strides=[1,1,1,1],
                  padding = 'VALID',
                  name='conv8_1x1')
    r8 = dtl.Relu(c8)
    
    c9 = dtl.Conv(r8,filter_shape=[1,1,nf2,10],
                  strides=[1,1,1,1],
                  padding='VALID',
                  name='conv9_1x1')
    r9 = dtl.Relu(c9)
    
    flat = dtl.Flatten(r9)
    
    nn = dtl.Network(x_image,[flat],bounds=[0.,1.])
    
    

    '''
    Note:
    	Initializing with only the last layer l3 will automatically construct the
        list of layers:

        dtl.Network(x_image,[l3],bounds=[0.,1.])
    '''
    
    y = nn.forward() #These are the logits
    #y=ro.forward() #Alternate

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    
    
    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    softmax = tf.nn.softmax_cross_entropy_with_logits(y, y_)
    cross_entropy = tf.reduce_mean(softmax)
    #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    
    with tf.Session() as sess:
        #init_op = tf.initialize_all_variables()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        saver = tf.train.Saver()
        #Test conv op by itself
            #batch_xs, batch_ys = mnist.train.next_batch(1)
            #tconv = sess.run(cl1.output,feed_dict = {x:batch_xs})
            #print ("Tconv",np.asarray(tconv).shape)

        if mode == "train":
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print ("Checkpoint restored")
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print ("No checkpoint found on ", ckpt)

            print "Running training mode..."
            #Instatiate SummaryWriter to output summaries and Graph (optional)
            #summary_writer = tf.train.SummaryWriter(summary_dir,sess.graph)
            summary_writer = tf.summary.FileWriter(summary_dir,sess.graph)


            for i in range(num_iterations):
                if i%2000 == 0:
                    print (i,"training iterations passed")
                batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
                #cl1_val = sess.run(cl1.output,feed_dict={x: batch_xs, y_: batch_ys})
                #print (np.asarray(cl1_val).shape)
                # Test trained model
                
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

            #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            #print(sess.run(accuracy, feed_dict={x: mnist.test.images,
            #                              y_: mnist.test.labels}))
            eval_model(sess,correct_prediction,mnist,x,y_)

            ckpt_name = "model_ckpt"
            save_path =saver.save(sess,checkpoint_dir+os.sep+ckpt_name)
            print("Model saved in file: %s" % save_path)
    
            
        #Decomposition:
        #Look up and retrieve most recent checkpoint
        elif mode =="visualize":
            print "Checkpoint dir", checkpoint_dir
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print "Checkpoint restored"
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print "No checkpoint found on", ckpt.model_checkpoint_path

        

            for i in range(num_visualize):
                batch_xs, batch_ys = mnist.train.next_batch(1)


                r_input = sess.run(nn.relevance_backprop(y*y_),
                                   feed_dict={x:batch_xs,y_:batch_ys})

                r_input_img=np.squeeze(r_input) 
                #r_input_img = np.reshape(r_input,(28,28))
                


                #utils.visualize(r_input[:,2:-2,2:-2],utils.heatmap,'deeptaylortest_'+str(i)+'_.png')

                
                #Display original input
                #plt.imshow(np.reshape(np.asarray(batch_xs),(28,28)))

                yguess_mat = sess.run(y,
                                 feed_dict={x: batch_xs, y_: batch_ys})
                yguess = yguess_mat.tolist()
                yguess = yguess[0].index(max(yguess[0]))
                actual = batch_ys[0].tolist().index(1.)

                print ("Guess:",(yguess))
                print ("Actual:",(actual))


                #Display relevance heatmap
                fig,ax = plt.subplots()
                im = ax.imshow(r_input_img,cmap=plt.cm.Reds,interpolation='nearest')
                ax.format_coord = Formatter(im)
                plt.show()
                

def eval_model(sess,correct_prediction,mnist,x,y_):

    eval_batch_size=1
    num_examples = mnist.test.images.shape[0]
    #num_labels = mnist.test.labels.shape[0]
    steps_per_epoch = num_examples//eval_batch_size

    num_correct=0
    for _ in range(steps_per_epoch):
        batch_xs, batch_ys = mnist.train.next_batch(eval_batch_size)
        feed_dict={x: batch_xs, y_: batch_ys}
        num_correct += sess.run(correct_prediction,feed_dict=feed_dict)

    precision = float(num_correct)/num_examples
    #summary_writer.add_summary(loss+'/'+summar_tag,step)
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, num_correct, precision))

            
if __name__ == '__main__':
    tf.app.run()
