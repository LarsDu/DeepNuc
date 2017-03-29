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


from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

#import utils

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string('data_dir',"/tmp/data","""MNIST data directory""")
flags.DEFINE_string('mode','train',"""Options: \'train\' or \'visualize\' """)

flags.DEFINE_integer('num_iterations',1000000,""" Number of training iterations """)
flags.DEFINE_integer('num_visualize',10,""" Number of samples to visualize""")


def main(_):
   
    mode = FLAGS.mode
    num_visualize = FLAGS.num_visualize
    num_iterations = FLAGS.num_iterations
    
    base_dir = "./mixed_test"
    checkpoint_dir = base_dir+"/checkpoints"
    summary_dir =base_dir+"/summary_dir"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    

    
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    x_image = ImageInput(x,image_shape=[28,28,1])
    


    
    #conv filter dimensions are  w,h,input_dims,output_dims
    
    
    #Pure Conv Test
    cl1 = ConvLayer(x_image, filter_shape = [5,5,1,10],padding = 'VALID',name='conv1')
    r1 = ReluLayer(cl1)
    p1 = AvgPoolLayer(r1,2,'avg_pool1')
    cl2 = ConvLayer(p1, filter_shape = [5,5,10,25],padding='VALID',name='conv2')
    r2 = ReluLayer(cl2)
    p2 = AvgPoolLayer(r2,2,'avg_pool2')
    flat = Flatten(p2)
    l1 = LinearLayer(flat,100,'linear1')
    r3 = ReluLayer(l1)
    l2 = LinearLayer(r3,50,'linear2' )  
    r4 = ReluLayer(l2)
    l3 = LinearLayer(r4,10,'linear3' )
    #cl3 = ConvLayer(p2,filter_shape = [4,4,25,100],padding='VALID',name='conv3')
    #r3 = ReluLayer(cl3)
    #p3 = AvgPoolLayer(r3,'avg_pool3')
    #cl4 = ConvLayer(p3,filter_shape=[1,1,100,10],padding='VALID',name='conv4')
    #flat = Flatten(cl4)
    
    nn = Network([cl1,r1,p1, cl2,r2,p2,flat,l1,r3,l2,r4,l3],bounds=[0.,1.])

   
    y = nn.forward() #These are the logits
    #y=ro.forward() #Alternate
    
    #y=nn.forward()
    #y_relu = tf.nn.relu(y)
    
    #grad_dy_dx = tf.gradients(y,x)
    #sqr_sensitivity = tf.pow(grad_dy_dx[0],2)
    
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
        init_op = tf.initialize_all_variables()
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
            summary_writer = tf.train.SummaryWriter(summary_dir,
                                                            sess.graph)


            for i in range(num_iterations):
                if i%2000 == 0:
                    print (i,"training iterations passed")
                batch_xs, batch_ys = mnist.train.next_batch(25)
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
                #cl1_val = sess.run(cl1.output,feed_dict={x: batch_xs, y_: batch_ys})
                #print (np.asarray(cl1_val).shape)
                # Test trained model
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                          y_: mnist.test.labels}))


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

                #Display relevance heatmap
                plt.imshow(r_input_img,cmap=plt.cm.Reds,interpolation='nearest')

                #Display original input
                #plt.imshow(np.reshape(np.asarray(batch_xs),(28,28)))

                


                yguess_mat = sess.run(y,
                                 feed_dict={x: batch_xs, y_: batch_ys})
                yguess = yguess_mat.tolist()
                yguess = yguess[0].index(max(yguess[0]))
                actual = batch_ys[0].tolist().index(1.)

                print ("Guess:",(yguess))
                print ("Actual:",(actual))
                
                plt.show()
                
                #gradient=np.squeeze(np.asarray(sess.run(sqr_sensitivity,feed_dict={x:batch_xs})))
                #gradient = np.reshape(gradient,(28,28))
                #print (gradient.shape)
                #plt.imshow(gradient,cmap=plt.cm.Reds,interpolation='nearest')
                #plt.show()

    


class ImageInput:
    def __init__(self,input,image_shape = [28,28,1], pad_size=2, name=''):
        self.input = input 
        self.input_shape = self.input.get_shape().as_list()
        self.image_shape = image_shape
        self.pad_size = pad_size
        self.name = name
        
        #batch,w,h,channels for conv input
        self.image_4d = tf.reshape(self.input,
                          [-1,image_shape[0],image_shape[1],image_shape[2]]) 
        self.paddings = tf.constant([[0,0],[self.pad_size,self.pad_size],
                                            [self.pad_size,self.pad_size],
                                             [0,0]])
        self.image_4d = tf.pad( self.image_4d,self.paddings,"CONSTANT")

        self.output=self.image_4d
        self.output_shape = self.output.get_shape().as_list()

        print "Image",self.name,"input shape:",self.input_shape
        print "Image",self.name,"output_shape:",self.output_shape

        
    
            
class Network:
    def __init__(self,layers,bounds=[0.,1.]):
        self.layers = layers #this should be a list of layers
        self.lowest = float(bounds[0])
        self.highest = float(bounds[1])
        
    def forward(self):
        """
        For forward prop of the network, it suffices to simply get the
        output of the very last layer. This is because the graph should
        already be fully defined.
        """
        return self.layers[-1].output
    
    def relevance_backprop(self,final_rj):
        num_layers = len(self.layers)
        first_layer_ind = num_layers-1
        #Propogate relevance from the last layer to the input layer
        prev_layer_rj = final_rj
        for i,layer in enumerate(self.layers[::-1]):
            #note that i will always count from 0 to num_layers-1
            
            if i == first_layer_ind:
                print "First layer detected",layer.name
                #Set higher and lower bounds
                layer.set_input_bounds(self.lowest,self.highest)
                prev_layer_rj = layer.relevance_backprop_zbeta(prev_layer_rj)
            else:
                prev_layer_rj = layer.relevance_backprop_zplus(prev_layer_rj)
                
        input_x_relevance = prev_layer_rj
        return input_x_relevance
               


                
class LinearLayer:
    def __init__(self,input_layer,output_size,name):

        """
        Note: If the input of this layer is 2D, you must precede this layer
        with a Flatten layer.
        """
        

        self.input_layer = input_layer
        self.X = self.input_layer.output
        self.input_shape = self.input_layer.output.get_shape().as_list()
        self.name = name

        #These values can be reset
        self.lowest = 0.
        self.highest= 1.
        
        
                      
        self.input_size = self.input_shape[1]
        self.output_size = int(output_size)


        #Note weights and biases must be tf.Variables(). Do not call tf.truncated_normal directly!
        self.W = init_weights(self.name+'_weights',shape=[self.input_size,self.output_size])
        self.B = init_bias(self.name+'_bias',shape=[self.output_size])

        self.output = self.forward()

        print ("Linear layer",self.name,"input shape:",self.input_shape)
        print ("Linear layer",self.name,"output len",str(self.output_size))
        
    def set_input_bounds(self,low_val,high_val):
        #LinearLayer
        self.lowest=float(low_val)
        self.highest=float(high_val)
        
        
        
    def forward(self):
        X = self.X
        y = tf.matmul(X,self.W)+self.B
        return y

    def backward(self,DY,W):
        DX = tf.matmul(DY,tf.transpose(W))
        return DX
    
    def relevance_backprop_zplus(self,Rj):
        #Linear Layer
        X = self.X
        #print (self.X.get_shape()[0])
        V = tf.maximum(0.,self.W)
        Z = tf.matmul(X,V)+1e-9
        S = tf.div(Rj,Z)
        C = tf.matmul(S,tf.transpose(V)) # S by maxed weights
        #xfi = tf.mul(self.X,C)
        Ri = tf.mul(X, C)

        return Ri
        

    def relevance_backprop_zbeta(self,Rj):
        #Linear Layer
        """
        The zbeta rule is just the zplus rule applied to a closed
         interval where l_i <= x_i <= h_i
        It should only be applied to the first layer of the network
         if the inputs are bounded. For instance, pixel data may be bounded
         to values between 0 and 255
        """
        #Note: some math operations in tensorflow can use overloaded operator notations:
        # see here:http://stackoverflow.com/questions/35094899/tensorflow-operator-overloading
        W = self.W
        X = self.X

        L = tf.ones_like(self.X,dtype=tf.float32)*self.lowest
        H = tf.ones_like(self.X,dtype=tf.float32)*self.highest
        W_max = tf.maximum(0.,W) # alternatively tf.maximum(0,self.W)
        W_min = tf.minimum(0.,W)
        
        #L and H should be matrices with the same dims as self.x
        #x_shape = self.x.get_shape()[1] #Note dim0 will be ? if self.x is placeholder
        Z = tf.matmul(X,self.W)-tf.matmul(L,W_max) - tf.matmul(H,W_min) + 1e-9
        S =tf.div( Rj,Z)
        Ri = X*tf.matmul(S,tf.transpose(W))-L*tf.matmul(S,tf.transpose(V))-H*tf.matmul(S,tf.transpose(U))
        return Ri

    
    
    
    
           

class ReluLayer:

    def __init__(self,input_layer,name=''):
        self.input_layer = input_layer
        self.X = input_layer.output
        self.output = self.forward()
        self.input_shape = self.input_layer.output.get_shape().as_list()
        self.output_shape = self.output.get_shape().as_list()
        self.name = name
        #flat_dims_sep_channels = [-1, self.output_shape[1]*self.output_shape[2], self.output_shape[3]]
        #flat_dims = [-1,self.output_shape[1]*self.output_shape[2]*self.output_shape[3]]
        #self.flat_output_sep_channels = tf.reshape(self.output, flat_dims_sep_channels)        
        #self.flat_output = tf.reshape(self.output, flat_dims)        
        self.relu_mask = tf.greater(self.X,0) #logical matrix that is 1 where X > 0

        print "Relu layer",self.name,"input shape:",self.input_shape
        print "Relu layer",self.name,"output shape",self.output_shape
        
    def forward(self):
        return tf.nn.relu(self.X)

    def backward(self,DY):
        #Return DX for positive layer inputs only
        #(positive X inputs into layer, not positive DY inputs)
        DX = tf.mul(DY,self.relu_mask) 
        return DX 

    def relevance_backprop_zplus(self,Rj):
        #Just return the input
        return Rj
    
    def relevance_backprop_zbeta(self,Rj):
        #Just return the input
        return Rj



class Flatten:
    def __init__(self,input_layer,name=''):
        self.X = input_layer.output
        input_shape = input_layer.output.get_shape().as_list()
        self.input_shape =[-1 if i is None else i for i in input_shape]
        self.name = name
        
        self.output = self.forward()
        self.output_shape = self.output.get_shape().as_list()
        
        
        print "Input shape of flatten layer",self.name,"is",self.input_shape
        print "Output shape of flatten layer is",self.name,"is",self.output_shape
        
        
        #If the input shape is 4d (batch,w,h,channels)
        # set flag to make sure backprop will reshape Ri back into the input shape
        #if len(self.input_shape) > 2:
        #    self.do_reshape_input = True #Flag used to make sure relevance backprop performs reshape
        #else:
        #    self.do_reshape_input = False
        

    def forward(self):
        return tf.reshape (self.X,[-1,np.prod(self.input_shape[1::])])

    def relevance_backprop_zbeta(self,DY):
        return relevance_backprop_zplus(DY)
        
    def relevance_backprop_zplus(self,DY):
        return tf.reshape (DY,self.input_shape)


            
class ConvLayer:
    def __init__(self,input_layer,filter_shape,padding = 'VALID',name=''):

        self.X = input_layer.output
        self.filter_shape = filter_shape
        self.padding = padding
        self.name = name
        #self.name
        self.W = init_weights(self.name + '_weights',shape=self.filter_shape)
        self.B = init_bias(self.name + '_bias',shape=[self.filter_shape[3]])
        
        self.output = self.forward(self.X,self.W,self.B)
        #self.output_shape_op = tf.get_shape(self.output)

        
        
        self.input_shape = self.X.get_shape().as_list()
        output_shape = tensorshape_to_list(self.output.get_shape())
        self.output_shape = output_shape
        
        self.lowest=0.
        self.highest=1.
        
        print ("Conv input",self.name,"shape:",self.input_shape)
        print ("Conv filter",self.name,"shape:",self.filter_shape)
        print ("Conv output shape",self.output_shape)


    def set_input_bounds(self,low_val,high_val):
        """
        For setting input boundaries if zbeta rule is used
        """
        #ConvLayer
        self.lowest=float(low_val)
        self.highest=float(high_val)

          
        
    def forward(self,X,W,B):
        #print ("W shape:",self.W.get_shape())
        filtered =tf.nn.conv2d(X,W,strides=[1,1,1,1],padding=self.padding)
        return filtered + B 

       
    def backward(self,DY,W):
                
        batch_size = 1

        layer_input_shape = [batch_size,self.input_shape[1],
                                        self.input_shape[2],
                                        self.input_shape[3]]


        filtered = tf.nn.conv2d_transpose(DY,W,layer_input_shape,strides=[1,1,1,1],
                                          padding = self.padding)
                
        
        #print conv2d_transpose_output
        #print "Custom op output",conv2d_transpose_output.get_shape().as_list()
        return filtered
        

    
    def relevance_backprop_zplus(self,Rj):
        #Conv layer
        #First run pure convnet with 2d outputs and inputs as a test
        
        W_max = tf.maximum(0.,self.W)
        zero_bias = tf.zeros_like(self.B,dtype=tf.float32)
        Z = self.forward(self.X,W_max,zero_bias)
        S = tf.div(Rj,Z)
        C = self.backward(S,W_max)
        Ri = tf.mul(self.X,C)
    
        #print ("conv Z shape",Z.get_shape())
        #print ("conv Z reduced shape",Z.get_shape())
   
        return Ri
        
        
    

    def relevance_backprop_zbeta(self,Rj):
        #ConvLayer
        W = self.W
        W_min = tf.minimum(0.,self.W)
        W_max = tf.maximum(0.,self.W)
      

        L = tf.zeros_like(self.X,dtype=tf.float32)+self.lowest
        H = tf.zeros_like(self.X,dtype=tf.float32)+self.highest
        zero_bias = tf.zeros_like(self.B,dtype=tf.float32)

        #These forward ops need to be altered
        Z = self.forward(self.X,self.W,zero_bias)-self.forward(L,W_max,zero_bias)-self.forward(H,W_min,zero_bias)+1e-9
        
        S = tf.div(Rj,Z)
        
        Ri = self.X*self.backward(S,W)-H*self.backward(S,W_min)-L*self.backward(S,W_max)
        return Ri
        


class AvgPoolLayer:
    def __init__(self,input_layer,pool_size,name):

        self.pool_size=pool_size
        self.input_layer = input_layer
        self.X = input_layer.output
        self.name = name


        self.input_shape = self.X.get_shape().as_list()

        
        self.output = self.forward(self.X)
        self.output_shape = self.output.get_shape().as_list()
        self.gradient=tf.gradients(self.output,self.X)[0]
        
        #self.gradient = tf.gradients(self.output,self.X) 
        print ("AvgPool",self.name,"input shape",self.input_shape)
        print ("AvgPool",self.name,"output shape",self.output_shape)
        
                        
    def forward(self,X):
        return tf.nn.avg_pool(X, ksize=[1, self.pool_size, self.pool_size, 1],
                              strides=[1, self.pool_size, self.pool_size, 1], padding='SAME')

    def relevance_backprop_zplus(self,Rj):
        Z = self.forward(self.X)+1e-9
        
        S = tf.div(Rj,Z)
        C = self.backward(S)
        Ri = tf.mul(self.X,C)
        return Ri

    def relevance_backprop_zbeta(self,Rj):
        return self.relevance_backprop_zplus()

    
    def backward(self,DY):
        #DY should have the same dimensions as the pooled output
        # of this layer.
        # Performing a broadcasted multiplication by the gradient matrix (which should just be
        # a matrix of 0.25's (for pool size of 2x2), should yield DX
        
        
        #depooled_dy = tensor_block_expand(DY,self.pool_size)
        #DX = tf.mul(depooled_dy,self.gradient)
        #print "DY avg pool shape",DY.get_shape().as_list()
        #print "Depooled dy avg pool shape",depooled_dy.get_shape().as_list()
               
        depooled_dy = tf.mul(tensor_block_expand(DY,self.pool_size),self.gradient)
        DX = depooled_dy*float(1./(self.pool_size**2.))
        return DX


    
class MaxPoolLayer:
    def __init__(self,input_layer,pool_size,name):
        
        self.pool_size=pool_size
        self.input_layer = input_layer
        self.X = input_layer.output
        self.name = name


        self.input_shape = self.X.get_shape().as_list()

        self.output = self.forward(self.X)
        self.output_shape = self.output.get_shape().as_list()

        self.gradient=tf.gradients(self.output,self.X)[0]
        
        #self.gradient = tf.gradients(self.output,self.X) 
        print ("MaxPool",self.name,"input shape",self.input_shape)
        print ("MaxPool",self.name,"output shape",self.output_shape)

        
        

    def forward(self,X):
        return tf.nn.max_pool(X, ksize=[1, self.pool_size, self.pool_size, 1],
                            strides=[1, self.pool_size, self.pool_size, 1], padding='SAME')

    def relevance_backprop_zplus(self,Rj):
        Z = self.forward(self.X)+1e-9
        S = tf.div(Rj,Z)
        C = self.backward(S)
        Ri = tf.mul(self.X,C)
        return Ri

    def relevance_backprop_zbeta(self,Rj):
        return self.relevance_backprop_zplus()

    def backward(self,DY):
        ##MaxPool
        #self.gradient should be a logical matrix with ones for every max element
        #passed through forward
        depooled_dy = tf.mul(tensor_block_expand(DY,self.pool_size),self.gradient)
        return depooled_dy
    
    

#Convenience functions across classes

def init_weights(name,shape):
    #Initialize weight variables from a truncated normal distribution
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial,name=name)

def init_bias(name,shape):
    #Initialize bias variables with a value of 0.1
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)



def tensorshape_to_list(tensor_shape):
#Converts tensorshape to list, with variable dimensions (?) replaced with -1
    out_list = []
    for dim in tensor_shape:
        #print (dim)
        #print (type(dim))
        if str(dim) == '?':
            dim = -1
        else:
            dim = int(dim)
        out_list.append(dim)

    return out_list



        
def tensor_block_expand(value,rep):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, rep*d0, rep*d1, ..., rep*dn, ch]

    I modified a code example from:https://github.com/tensorflow/tensorflow/issues/2169
    """
    
    shape = value.get_shape().as_list()
    num_dims = len(shape[1:-1]) #Num interior dims (d0,d1,d2,dn...)
    #For [b,d0,d1,ch], reshape to [b*d0,d1,ch]
    out = (tf.reshape(value, [-1] + shape[-num_dims:]))

    for i in range(num_dims,0,-1): 
        out = tf.concat(i, [out]*rep)

    out_size = [-1] + [s*rep for s in shape[1:-1]] + [shape[-1]]
    out = tf.reshape(out, out_size)
    return out

       
    


    
        
if __name__ == '__main__':

    tf.app.run()




         
    
