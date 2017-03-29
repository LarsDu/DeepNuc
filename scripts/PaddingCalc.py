import numpy as np
import math

"""
Some simple functions for calculating the padding for "VALID"
and "SAME"

This is useful for calculating the hyperparameters for a fully
convolutional neural network
"""

def main():
    nf1=96
    nf2=48
    nf3 = 1024
    
    """
    #96x96 network calculations
     num_classes = 2
    c1 = output_shape_conv([25,96,96,4],[9,9,4,nf1],[1,1,1,1],'VALID')
    c2 = output_shape_conv(c1,[3,3,nf1,nf1],[1,2,2,1],'VALID') #strided conv 

    c3 = output_shape_conv(c2,[4,4,nf1,nf1],[1,1,1,1],'VALID')
    c4 = output_shape_conv(c3,[3,3,nf1,nf1],[1,2,2,1],'VALID')#strided conv 

    c5 = output_shape_conv(c4,[4,4,nf1,nf1],[1,1,1,1],'VALID')
    c6 = output_shape_conv(c5,[3,3,nf1,nf2],[1,2,2,1],'VALID') #7x7

    c7 = output_shape_conv(c6,[3,3,nf2,nf2],[1,1,1,1],'VALID') #7x7
    c8 = output_shape_conv(c7,[3,3,nf1,nf2],[1,2,2,1],'VALID') #7x7

    #fc style layers
    
    c9 = output_shape_conv(c8,[1,1,nf2,nf1],[1,1,1,1],'VALID')
    c10 = output_shape_conv(c9,[1,1,nf1,nf2],[1,1,1,1],'VALID')
    c11 = output_shape_conv(c10,[1,1,nf2,num_classes],[1,1,1,1],'VALID')
    """
    """
    #96x96 shallower with 5x5 layers using stride 2
    c1 = output_shape_conv([25,96,96,4],[9,9,4,nf1],[1,1,1,1],'VALID')
    c2 = output_shape_conv(c1,[5,5,nf1,nf1],[1,2,2,1],'VALID') #strided conv 

    c3 = output_shape_conv(c2,[5,5,nf1,nf1],[1,2,2,1],'VALID')
    c4 = output_shape_conv(c3,[5,5,nf1,nf1],[1,2,2,1],'VALID')#strided conv 

    c5 = output_shape_conv(c4,[5,5,nf1,nf3],[1,2,2,1],'VALID')
    
    """
    #Flatten output and pass 

    '''
    #96x96 really simply convolution with flat linear output
        #96x96 shallower with 5x5 layers using stride 2
    c1 = output_shape_conv([25,96,96,4],[9,9,4,nf1],[1,1,1,1],'VALID')
    c2 = output_shape_conv(c1,[5,5,nf1,nf1],[1,2,2,1],'VALID') #strided conv 

    c3 = output_shape_conv(c2,[5,5,nf1,nf1],[1,2,2,1],'VALID')
    c4 = output_shape_conv(c3,[5,5,nf1,nf1],[1,2,2,1],'VALID')
    c5 = output_shape_conv(c4,[5,5,nf1,nf1],[1,2,2,1],'VALID')
    '''
    """
    nf1=32
    nf2=64
    z=128

    #generatorA and discriminatorA architecture
    c1 = output_shape_conv([25,96,96,3],[9,9,3,nf2],[1,1,1,1],'SAME')#dc9
    c2 = output_shape_conv(c1,[5,5,nf2,nf1],[1,2,2,1],'SAME') #dc8
    c3 = output_shape_conv(c2,[5,5,nf1,nf1],[1,2,2,1],'SAME') #dc7
    c4 = output_shape_conv(c3,[5,5,nf1,nf1],[1,2,2,1],'SAME') #dc6
    c5 = output_shape_conv(c4,[5,5,nf2,nf2],[1,2,2,1],'SAME') #dc5
    c6 = output_shape_conv(c5,[5,5,nf2,nf2],[1,2,2,1],'SAME') #dc4
    c7 = output_shape_conv(c6,[3,3,nf2,nf2],[1,3,3,1],'SAME') #dc3
    c8 = output_shape_conv(c7,[1,1,nf2,nf3],[1,1,1,1],'SAME') #dc2
    c9 = output_shape_conv(c8,[1,1,nf3,z],[1,1,1,1],'SAME') #dc1
    """

    #InfC SAME padding model - Based on https://arxiv.org/pdf/1412.6806.pdf
    # But adapted to 1D sequences. All pooling layers replace by strided
    # convolution
    seq_len=600
    fw1 = 30
    fw2 = 26
    stride = 2
    nf1=96
    nf2 = 192

    
    c1 = output_shape_conv([1,1,seq_len,4],[1,fw1,4,nf1],[1,1,1,1],'SAME')
    c2 = output_shape_conv(c1,[1,fw1,nf1,nf1],[1,1,1,1],'SAME')
    c3 = output_shape_conv(c2,[1,fw1,nf1,nf1],[1,1,stride,1],'SAME') #strided
    
    c4 = output_shape_conv(c3,[1,fw2,nf1,nf2],[1,1,1,1],'SAME')
    c5 = output_shape_conv(c4,[1,fw2,nf2,nf2],[1,1,stride,1],'SAME')
    c6 = output_shape_conv(c5,[1,fw2,nf2,nf2],[1,1,stride,1],'SAME') #strided

    #c7 = output_shape_conv(c6,[1,fw2,nf2,nf2],[1,1,stride,1],'SAME')
    #c8 = output_shape_conv(c7,[1,fw2,nf2,nf2],[1,1,stride,1],'SAME')
    #c9s = output_shape_conv(c8,[1,fw2,72,72],[1,1,stride,1],'SAME') #strided

    
    
def output_shape_conv(input_shape,filter_shape,strides,padding,verbose = True):

    if padding == 'VALID':
        output_shape = calc_valid(input_shape,filter_shape,strides)
    elif padding == 'SAME':
        output_shape = calc_same(input_shape,filter_shape,strides)
    if verbose:
        print "Padding mode",padding
        print "Input shape",input_shape
        print "Filter shape",filter_shape
        print "Output shape", output_shape,"\n" 
    return output_shape


def calc_valid(input_shape,filter_shape,strides):
    #batch_size = input_dims[0]
    #num_input_channels = input_dims[-1]
    out_shape = input_shape[:]
    for i,_ in enumerate(input_shape[1:-1]):
        cur_dim = i+1
        out_shape[cur_dim] = int(math.ceil( float(input_shape[cur_dim]-filter_shape[cur_dim-1]+1))
                                        /float(strides[cur_dim]) )

    out_shape[-1] = filter_shape[-1] 
    return out_shape



        
def calc_same(input_shape,filter_shape,strides):
    #ref: https://www.tensorflow.org/api_docs/python/nn/convolution
    #batch_size = input_dims[0]
    #num_input_channels = input_dims[-1]
    out_shape = input_shape[:]
    for i,_ in enumerate(input_shape[1:-1]):
        cur_dim = i+1
        
        out_shape[cur_dim] = int(math.ceil( float(input_shape[cur_dim])/float(strides[cur_dim]) ))
        #pad_along_dim = ((out_shape[cur_dim]-1)*strides[cur_dim]+
        #                  filter_shape[cur_dim]-input_shape[cur_dim])
        #pad_dim = pad_along_dim/2 #top and bottom get additional pixel

    out_shape[-1] = filter_shape[-1] 
    return out_shape

    
    
if __name__ == "__main__":
    main()
