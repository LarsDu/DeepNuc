import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

"""
An implementation of Layer Wise Relevance Propagation in Tensorflow
with some other good stuff

Written by Lawrence Du
"""

class NucInput:
    def __init__(self,_input,pad_size,name='nuc_input'):
        #Nucleotide input should be assumed to be shape =[batch_size,4,seq_len]
        #reshape to [batch_size,height=1,width=seq_len,num_channels=4]
        
        self._input=_input
        self.input_shape = self._input.get_shape().as_list() #[b,4,seq_len]
        self.seq_len = self.input_shape[2]
        

        self.pad_size = pad_size
        #Paddings should only be applied to width dimension (dim[1])
        self.paddings = tf.constant([[0,0],[0,0],[self.pad_size,self.pad_size],[0,0]])

        #I recommend fiddling with PaddingCalc.py to decide on padding amount
        #when using 'VALID' mode. For nuc input of 600 bp with filter 25, a pad_size=12
        # will produce convolved output that is 600 elements wide
    
        
        
        #Apply paddings
        self.output = self.forward(self._input)
        self.output_shape = self.output.get_shape().as_list()

    def forward(self, input_data):
        #[b,4,seq_len]--transperm-->[b,seq_len,4]--expand_dim-->[b,1,seq_len,4]-->pad
        output = tf.expand_dims(tf.transpose(input_data, perm=[0,2,1]),1)
        return tf.pad(output,self.paddings,mode="CONSTANT")

    def backward(self,DY):
        return self.gradprop(DY)
    
    def gradprop(self,DY):
        #Useful for reshaping relevance backprop data back into the original
        # input dimensions. Basically just the reverse of the forward op

        #First depad by slicing
        #[b,1,seq_len_padded,4]-->[b,1,seq_len,4]
        #Note:-1 means include all elements in that dimension
        out = tf.slice(DY,begin=[0,0,self.pad_size,0],size=[-1,-1,self.seq_len,-1])

        #[b,1,seq_len,4]--squeeze-->[b,seq_len,4]--transperm-->[b,4,seq_len]

        DX = tf.transpose(tf.squeeze(out,[1]),perm=[0,2,1])
        return DX


        #Remove extraneous dims

        
        
    
class ImageInput:
    def __init__(self,input_data,image_shape = [28,28,1], pad_size=2, name='img_input'):
        self._input = input_data 
        self.input_shape = self._input.get_shape().as_list()
        self.image_shape = image_shape
        self.pad_size = pad_size
        self.name = name
        
        #batch,w,h,channels for conv input
        self.paddings = tf.constant([[0,0],[self.pad_size,self.pad_size],
                                            [self.pad_size,self.pad_size],
                                             [0,0]])
        

        self.output=self.forward(self._input)
        self.output_shape = self.output.get_shape().as_list()

        print "Image",self.name,"input shape:",self.input_shape
        print "Image",self.name,"output_shape:",self.output_shape,"\n"
        

    def forward(self,_input):
        image_4d = tf.reshape(self._input,
                      [-1,self.image_shape[0],self.image_shape[1],self.image_shape[2]]) 
        image_4d = tf.pad( image_4d,self.paddings,"CONSTANT")
        return image_4d

    def depad(self,DY):
        depad = tf.slice(DY,begin=[0,self.pad_size,self.pad_size,0],
                            size=[-1,self.image_shape[0],self.image_shape[1],-1])
        return depad

    def backward(self,DY):
        return self.gradprop(DY)
    
    def gradprop(self,DY):
        #Returns shape to [b,1d_image]
        return tf.reshape(self.depad(DY),[-1,self.input_shape[1]])
            
        
            
            
class Network:
    def __init__(self,_input,layers=None,bounds=[0.,1.]):
        #The input should contain a backward() function
        #to reverse any reshaping performed on the input
        self._input = _input


        #layers can be a list of layers or only the last layer in a network
        
        if len(layers) == 1:
            
            self.layers = layers[0].get_layers_list()
            #If only one input, construct the list of layers
        else:
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

    def input_relevance_backprop(self,final_rj):
        rel_backprop = self.relevance_backprop(final_rj)
        rel_backprop_deshaped = self._input.gradprop(rel_backprop)
        return rel_backprop_deshaped
      
               

    
class Layer:
    def get_layers_list(self):
        '''
        Get a list of all previous layers 
        by iterating layer by layer
        '''
        layers_list = []
        layers_list.append(self)
        layers_list.append(self.input_layer)
        has_input_layer = True
        while (has_input_layer):
            try:
                prev_layer = layers_list[-1].input_layer
                layers_list.append(prev_layer)
            except:
                has_input_layer = False
                layers_list.pop() # remove 'ImgInput or NucInput'
                

            
        return layers_list[::-1]


         
class Linear(Layer):
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

        print "Linear layer",self.name,"input shape:",self.input_shape
        print "Linear layer",self.name,"output len",str(self.output_size),"\n"
        
    def set_input_bounds(self,low_val,high_val):
        #Linear Layer
        self.lowest=float(low_val)
        self.highest=float(high_val)
        
        
        
    def forward(self):
        X = self.X
        y = tf.matmul(X,self.W)+self.B
        return y

    def gradprop(self,DY,W):
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

    
    
    
          

class Relu(Layer):
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
        print "Relu layer",self.name,"output shape",self.output_shape,"\n"
        
    def forward(self):
        return tf.nn.relu(self.X)

    def gradprop(self,DY):
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



class Flatten(Layer):
    def __init__(self,input_layer,name=''):
        self.input_layer = input_layer
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
        #print "Deflattening to", self.input_shape
        return tf.reshape (DY,self.input_shape)


            
class Conv2d(Layer):
    def __init__(self,
                 input_layer,
                 filter_shape,
                 strides = [1,1,1,1],
                 padding = 'VALID',
                 name=''):

        self.input_layer = input_layer
        self.X = input_layer.output
        self.filter_shape = filter_shape
        self.padding = padding
        self.strides = strides
        self.name = name
        #self.name
        self.W = init_weights(self.name + '_weights',shape=self.filter_shape)
        self.B = init_bias(self.name + '_bias',shape=[self.filter_shape[3]])
        
        self.output = self.forward(self.X,self.W,self.B)
        #self.output_shape_op = tf.get_shape(self.output)

        
        
        self.input_shape = self.X.get_shape().as_list()
        output_shape = self.output.get_shape().as_list()
        self.output_shape = output_shape
        
        self.lowest=0.
        self.highest=1.
        
        print "Conv input",self.name,"shape:",self.input_shape
        print "Conv filter",self.name,"shape:",self.filter_shape
        print "Conv output shape",self.output_shape,"\n"


    def set_input_bounds(self,low_val,high_val):
        """
        For setting input boundaries if zbeta rule is used
        Unlike other params, these values can be set after
        graph initialization and training
        """
        #Conv Layer
        self.lowest=float(low_val)
        self.highest=float(high_val)

          
        
    def forward(self,X,W,B):
        #Conv Layer
        with tf.variable_scope(self.name+'_forward') as scope:
            #print ("W shape:",self.W.get_shape())
            out =tf.nn.conv2d(X,W,strides=self.strides,padding=self.padding)+B
        activation_summary(out)
        return out

       
    def gradprop(self,DY,W):
        #Note: for now, the batch size for backward relevance propagation must be set
        # to 1

        #TODO: Make this op work for batch_size>1
        #print "Conv back",DY.get_shape().as_list()
        batch_size = 1
        layer_input_shape = [batch_size,self.input_shape[1],
                                        self.input_shape[2],
                                        self.input_shape[3]]

        filtered = tf.nn.conv2d_transpose(DY,W,layer_input_shape,
                                          strides=self.strides,
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
        C = self.gradprop(S,W_max)
        Ri = tf.mul(self.X,C)
    
        #print ("conv Z shape",Z.get_shape())
        #print ("conv Z reduced shape",Z.get_shape())
        return Ri
        
        
    

    def relevance_backprop_zbeta(self,Rj):
        #Conv Layer
        W = self.W
        W_min = tf.minimum(0.,self.W)
        W_max = tf.maximum(0.,self.W)
      

        L = tf.zeros_like(self.X,dtype=tf.float32)+self.lowest
        H = tf.zeros_like(self.X,dtype=tf.float32)+self.highest
        zero_bias = tf.zeros_like(self.B,dtype=tf.float32)

        #These forward ops need to be altered
        Z = self.forward(self.X,self.W,zero_bias)-self.forward(L,W_max,zero_bias)-self.forward(H,W_min,zero_bias)+1e-9
        S = tf.div(Rj,Z)
        Ri = self.X*self.gradprop(S,W)-H*self.gradprop(S,W_min)-L*self.gradprop(S,W_max)
        return Ri
        


class AvgPool(Layer):

    def __init__(self,input_layer,pool_dims,name):
        #Note for now, I will keep this op using 'SAME' padding
        #Padding can complicate depooling:
        # An explanation can be found here:
        #http://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
        self.pool_dims = pool_dims
        self.input_layer = input_layer
        self.X = input_layer.output
        self.name = name
        self.strides = [1, self.pool_dims[0], self.pool_dims[1], 1]
        self.k_size= [1, self.pool_dims[0], self.pool_dims[1],1]
        self.input_shape = self.X.get_shape().as_list()

        
        self.output = self.forward(self.X)
        self.output_shape = self.output.get_shape().as_list()
        self.gradient=tf.gradients(self.output,self.X)[0]

        #Padding calculations for 'SAME' padding
        #Used for correcting depooling

        out_height = np.ceil( self.input_shape[1]/np.float32(self.strides[1]))
        out_width = np.ceil( self.input_shape[2]/np.float32(self.strides[2]))
        self.pad_along_height =((out_height-1)*
                                self.strides[1]+
                                self.k_size[1]-self.input_shape[1])
                                 
        self.pad_along_width = ((out_width-1)*
                                self.strides[2]+
                                self.k_size[2]-self.input_shape[2])

        #Bottom and right sides always take additional pixel
        self.pad_bottom = np.ceil(self.pad_along_height/2.) 
        self.pad_top = self.pad_along_height-self.pad_bottom
        self.pad_right = np.ceil(self.pad_along_width/2.)
        self.pad_left = self.pad_along_width-self.pad_right
        

        
        #self.gradient = tf.gradients(self.output,self.X) 
        print "AvgPool",self.name,"input shape",self.input_shape
        print "AvgPool",self.name,"output shape",self.output_shape
        print ("'SAME' padding: top -",self.pad_top,
                               "bottom -",self.pad_bottom,
                               "left -",self.pad_left,
                               "right -",self.pad_right)
        print "\n"
                        
    def forward(self,X):
        #AvgPool Layer
        with tf.variable_scope(self.name+'_forward') as scope:

            out= tf.nn.avg_pool(X, ksize=self.k_size,
                              strides=self.strides, padding='SAME')
            activation_summary(out)
        return out
        
    def relevance_backprop_zplus(self,Rj):
        #AvgPool Layer
        with tf.variable_scope(self.name+'_backward_zplus') as scope:
            Z = self.forward(self.X)+1e-9
            S = tf.div(Rj,Z)
            C = self.gradprop(S)
            Ri = tf.mul(self.X,C)
            #TODO: Troubleshoot this section
            #X is [b,1,128,48] y is [b,32,4,48]
            #B should be [b,1,128,48]
            #activaton_summary(Ri)
        return Ri

    def relevance_backprop_zbeta(self,Rj):
        return self.relevance_backprop_zplus()

    
    def gradprop(self,DY):
        """
        DY should have the same dimensions as the pooled output

        #Steps:
        1. Depool by expanding rows and columns in height and width dims
        2. Remove any padding added to original input 
        3. Multiply by gradient
             For Maxpooling, gradient is a binary matrix with ones for each max item
             For Avg pooling, gradient should be a matrix of 1./np.prod(self.pool_dims)
        
        """
        #Depool
        depooled_dy = depool_2d(DY,self.pool_dims)
        #Slice out padding
        depadded = tf.slice(depooled_dy,
                            begin=[0,int(self.pad_top),int(self.pad_right),0],
                            size=[-1,self.input_shape[1],self.input_shape[2],-1])
        #print "Depooled size",depooled_dy.get_shape().as_list()
        #print "Pooling gradient dims", self.gradient.get_shape().as_list()

        DX = tf.mul(depadded,self.gradient)
        #DX = tf.mul(depooled_dy,float(1./np.prod(self.pool_dims)))
        return DX


    
class MaxPool(Layer):
    def __init__(self,input_layer,pool_dims,name):

        self.pool_size=pool_size
        self.input_layer = input_layer
        self.X = input_layer.output
        self.name = name
        self.pool_dims = pool_dims
        self.strides = [1, self.pool_dims[0], self.pool_dims[1], 1]
        self.k_size= [1, self.pool_dims[0], self.pool_dims[1],1]
        self.input_shape = self.X.get_shape().as_list()

        self.output = self.forward(self.X)
        self.output_shape = self.output.get_shape().as_list()

        #Padding calculations for 'SAME' padding
        #Used for correcting depooling

        out_height = np.ceil( self.input_shape[1]/np.float32(self.strides[1]))
        out_width = np.ceil( self.input_shape[2]/np.float32(self.strides[2]))
        self.pad_along_height =((out_height-1)*
                                self.strides[1]+
                                self.k_size[1]-self.input_shape[1])
                                 
        self.pad_along_width = ((out_width-1)*
                                self.strides[2]+
                                self.k_size[2]-self.input_shape[2])


        self.pad_bottom = np.ceil(self.pad_along_height/2.) 
        self.pad_top = self.pad_along_height-self.pad_bottom
        self.pad_right = np.ceil(self.pad_along_width/2.)
        self.pad_left = self.pad_along_width-self.pad_right
        

        #self.gradient = tf.gradients(self.output,self.X) 
        print "MaxPool",self.name,"input shape",self.input_shape
        print "MaxPool",self.name,"output shape",self.output_shape
        print ("'SAME' padding: top -",self.pad_top,
                               "bottom -",self.pad_bottom,
                               "left -",self.pad_left,
                               "right -",self.pad_right)
        print "\n"
        

    def forward(self,X):
        #MaxPool Layer
        with tf.variable_scope(self.name+'_forward') as scope:
            out = tf.nn.max_pool(X, ksize=self.ksize,
                            strides=self.strides, padding='SAME')
            activation_summary(out)
        return out

    def relevance_backprop_zplus(self,Rj):
        with tf.variable_scope(self.name+'backward_zplus') as scope:
            Z = self.forward(self.X)+1e-9
            S = tf.div(Rj,Z)
            C = self.gradprop(S)
            Ri = tf.mul(self.X,C)
            #activaton_summary(Ri)
        return Ri

    def relevance_backprop_zbeta(self,Rj):
        return self.relevance_backprop_zplus()

    def gradprop(self,DY):
        """
        DY should have the same dimensions as the pooled output

        #Steps:
        1. Depool by expanding rows and columns in height and width dims
        2. Remove any padding added to original input 
        3. Multiply by gradient
             For Maxpooling, gradient is a binary matrix with ones for each max item
             For Avg pooling, gradient should be a matrix of 1./np.prod(self.pool_dims)
        
        """
        #Depool
        depooled_dy = depool_2d(DY,self.pool_dims)
        #Slice out padding
        depadded = tf.slice(depooled_dy,
                            begin=[0,int(self.pad_top),int(self.pad_right),0],
                            size=[-1,self.input_shape[1],self.input_shape[2],-1])
        #print "Depooled size",depooled_dy.get_shape().as_list()
        #print "Pooling gradient dims", self.gradient.get_shape().as_list()

        DX = tf.mul(depadded,self.gradient)
        #DX = tf.mul(depooled_dy,float(1./np.prod(self.pool_dims)))
        return DX


class BatchNorm(Layer):
    #ref: http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
    #ref  http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    #ref: https://github.com/tensorflow/tensorflow/blob/b826b79718e3e93148c3545e7aa3f90891744cc0/tensorflow/contrib/layers/python/layers/layers.py#L100

    def __init__(self,input_layer):
        pass

    def batch_norm():
        pass

    def forward():
        pass

    def gradprop(self,DY):
        #Do absolutely nothing to input
        return DY
    
    
    def relevance_backprop_zplus(self,Rj):
        #Just return the input
        return Rj
    
    def relevance_backprop_zbeta(self,Rj):
        #Just return the input
        return Rj

    	
class Dropout(Layer):
    def __init__(self,input_layer,keep_prob,name = 'dropout'):
        self.input_layer = input_layer
        self.keep_prob = keep_prob
        self.output = self.forward()				
        self.name = name

        self.input_shape = self.input_layer.output.get_shape().as_list()
        self.output_shape = self.output.get_shape().as_list()
        print "Dropout",self.name,"input shape:",self.input_shape
        print "Dropout",self.name,"output shape",self.output_shape,"\n"
        	
    def forward (self):
        return tf.nn.dropout(self.input_layer.output,self.keep_prob)
		
    def gradprop(self,DY):
        #Do absolutely nothing to input
        return DY
    
    
    def relevance_backprop_zplus(self,Rj):
        #Just return the input
        return Rj
    
    def relevance_backprop_zbeta(self,Rj):
        #Just return the input
        return Rj


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



 
def depool_2d(value,rep_list):
    """
    Duplicates the interior dimensions (rows and columns) of a 4d tensor
    where the first and last dimensions are batch_size and num channels
    :param value: A Tensor of shape [b, d0, d1,h]
    :return: A Tensor of shape [b, rep[0]*d0, rep[1]*d1, ch]
    
    
    """

    
    
    shape = value.get_shape().as_list()
    inner_dims = shape[1:-1]
    
    if len(inner_dims) != 2:
        print ("Number of non-batch/non-channel dimensions of input",
               "does not match rep_list")
        return None

    #batch_size = shape[0] #this should be None
    num_channels = shape[-1]
    #out = (tf.reshape(value, [-1] + shape[-num_dims:]))

    #Duplicate each column rep_list[1] times
    out = tf.reshape(value,[-1,inner_dims[0]*inner_dims[1],1,num_channels])
    out = tf.tile(out,[1,1,rep_list[1],1])
    out = tf.reshape(out,[-1,inner_dims[0],inner_dims[1]*rep_list[1],num_channels])

    #Duplicate each row rep_list[0] times

    out = tf.tile(out,[1,1,rep_list[0],1])
    final_dims = [-1,inner_dims[0]*rep_list[0],inner_dims[1]*rep_list[1],num_channels]
    out = tf.reshape(out,final_dims)
        
    
    return out



    




def activation_summary(in_op):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
    x: Tensor
    Returns:
    nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    #tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tensor_name = in_op.op.name
    tf.summary.histogram(tensor_name + '/activations', in_op)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(in_op))

    
        
if __name__ == '__main__':

    tf.app.run()




         
    
