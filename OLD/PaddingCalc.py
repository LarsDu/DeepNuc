import numpy as np
import math

"""
Some simple functions for calculating the padding for "VALID"
and "SAME"

This is useful for calculating the hyperparameters for a fully
convolutional neural network
"""

def main():

    #Testing a fully convolutional network for a sequence 600 bp long
    """InfB model
    c1s = output_shape_conv([1,1,624,4],[1,20,4,20],[1,1,1,1],'VALID')
    a1s = output_shape_conv(c1s,[1,1,4,1],[1,1,4,1],'SAME')
    c2s = output_shape_conv(a1s,[1,15,20,100],[1,1,1,1],'VALID')
    a2s = output_shape_conv(c2s,[1,1,4,1],[1,1,4,1],'SAME')
    c3s = output_shape_conv(a2s,[1,8,100,24],[1,1,1,1],'VALID')
    a3s = output_shape_conv(c3s,[1,1,4,1],[1,1,4,1],'SAME')
    c4s = output_shape_conv(a3s,[1,6,24,1],[1,1,1,1],'VALID')

    #output_shape_conv([3,5,10,4],[5,6,10,20],[1,1,1,1],'SAME')
    #output_shape_conv([3,32,32,4],[5,5,10,20],[1,1,1,1],'VALID')
    #layer2_shape = output_shape_conv([3,800,6],[24,10,20],[1,1,1],'VALID')
    #output_shape_conv(layer2_shape,[24,10,20],[1,1,1],'VALID')
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

    
    '''
    #InfC model - Based on https://arxiv.org/pdf/1412.6806.pdf
    # But adapted to 1D sequences. All pooling layers replace by strided
    # convolution
    c1 = output_shape_conv([1,1,624,4],[1,30,4,96],[1,1,1,1],'VALID')
    c2 = output_shape_conv(c1,[1,30,96,96],[1,1,1,1],'VALID')
    c3s = output_shape_conv(c2,[1,30,96,96],[1,1,4,1],'VALID') #strided
    
    
    c4 = output_shape_conv(c3s,[1,25,96,192],[1,1,1,1],'VALID')
    c5 = output_shape_conv(c4,[1,25,192,192],[1,1,1,1],'VALID')
    c6s = output_shape_conv(c5,[1,25,192,72],[1,1,4,1],'VALID') #strided

    c7 = output_shape_conv(c6s,[1,15,72,72],[1,1,1,1],'VALID')
    c8 = output_shape_conv(c7,[1,1,72,2],[1,1,1,1],'VALID')
    #c9s = output_shape_conv(c8,[1,25,72,72],[1,1,2,1],'VALID') #strided
    '''

    #InfD model - A version of InfC with a fully connected layer at the end

    
    """
    #All CNN-C
    c1 = output_shape_conv([1,32,32,3],[3,3,3,96],[1,1,1,1],'VALID')
    c2 = output_shape_conv(c1,[3,3,96,96],[1,1,1,1],'VALID')
    c3s = output_shape_conv(c2,[3,3,96,96],[1,2,2,1],'VALID') #strided
    
    
    c4 = output_shape_conv(c3s,[3,3,96,192],[1,1,1,1],'VALID')
    c5 = output_shape_conv(c4,[3,3,192,192],[1,1,1,1],'VALID')
    c6s = output_shape_conv(c5,[3,3,192,192],[1,2,2,1],'VALID') #strided

    c7 = output_shape_conv(c6s,[3,3,192,192],[1,1,1,1],'VALID')
    c8 = output_shape_conv(c7,[1,1,192,192],[1,1,1,1],'VALID')
    c9 = output_shape_conv(c8,[1,1,192,10],[1,1,1,1],'VALID')
    # In the original paper, this is followed by global averaging over 6x6
    # spatial dims followed by 10 or 100 way softmax
   """

    
def output_shape_conv(input_shape,filter_shape,strides,padding,verbose = True):

    if padding == 'VALID':
        output_shape = calc_valid(input_shape,filter_shape,strides)
    elif padding == 'SAME':
        output_shape = calc_same(input_shape,strides)
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



        
def calc_same(input_shape,strides):
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

    #out_shape[-1] = filter_shape[-1] 
    return out_shape

    
    
if __name__ == "__main__":
    main()
