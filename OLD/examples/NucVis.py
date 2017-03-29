import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
import numpy as np
from duseqlogo.LogoTools import *
import os




def vis_conv_filters(conv_filter,output_image):
    #num_filters =  ntconv.FLAGS.num_filters
    num_filters = conv_filter.shape[3]
    filt_list  = []
    for i in range(num_filters):
        #sigmoid = lambda(x): 1./(1.+np.exp(-x))
        #softmax = lambda(x): np.exp(x)/np.sum(np.exp(x),axis=0)
        #Shape is (1,conv_filter_width,4,NUM_FILTERS)
        #Correct conversion of conv filter to probs.
        raw_filter = conv_filter[0,:,:,i].T 
        relu_filter = raw_filter*(raw_filter>0)
        #norm_relu_filter = relu_filter/np.clip(np.sum(relu_filter,axis=0),1e-10,1000)
        #norm_relu_filter = relu_filter
        #norm_relu_filter=np.power(norm_relu_filter,2)
        #print 'Filter'
        #print norm_relu_filter
        #pprint.pprint(raw_filter)
        filt_list.append(relu_filter)
        logo_sheet = LogoSheet(filt_list,'conv',is_onehot_4d=True)
        #logo_sheet = LogoSheet(filt_list,'pwm')
        logo_sheet.draw_conv_filter()
        #logo_sheet.draw_pwm()
        logo_sheet.write_to_png(output_image)

def vis_filter_resp(sess,
                    data_placeholder,
                    labels_placeholder,
                    keep_prob_placeholder,
                    nuc_conv_model,
                    nuc_data,
                    save_dir = '.'):
    #Visit every example from original data set
    #Input is a numpy ndarray with shape:
    #[batch_size,self.seq_len, self.num_filters]
    data = nuc_data.all_data
    labels = nuc_data.all_labels
    
    num_examples = data.shape[0]
    seq_len = data.shape[2]
    #print 'seq_len',seq_len
    num_filters = nuc_conv_model.num_filters
    all_h_conv = np.zeros((seq_len,num_filters)) 
    for i in range(num_examples):
        
        all_h_conv += np.squeeze(sess.run(nuc_conv_model.h_conv1,feed_dict=
                  {data_placeholder:[data[i]],
                    labels_placeholder:[labels[i]],
                    keep_prob_placeholder:1.0}))

    #After squeezing, each h_conv should have shape [seq_len,num_filters]    
    #Element-wise divide to get an average sequence response
    #across entire data set.
    avg_h_conv = np.divide(all_h_conv,num_examples)
    print 'avg_h_conv shape ',avg_h_conv.shape
    seq_len = avg_h_conv.shape[0]
    num_filters = avg_h_conv.shape[1]

    #Draw each column (corresponding to a filter)
    for col in range(num_filters):
        avg_filt_resp = avg_h_conv[:,col]
        plt.figure()
        plt.xlabel('Position relative to TSS')
        plt.ylabel('Average motif filter response')
        plt.plot(range(80-seq_len//2, (seq_len-80)-seq_len//2),
                 avg_filt_resp[80:seq_len-80])
        plt.axis([int(-(seq_len-160)//2), int((seq_len-160)//2), 0, .45])
        #n,bins,_ = plt.hist(avg_filt_resp,
        #                    bins=seq_len,
        #                    normed=True,
        #                    facecolor='blue',
        #                    alpha=1.0)
        #plt.plot(bins[1:],n)
        
        #plt.show()
        plt.savefig(save_dir+os.sep+'filt_'+str(col)+'_test.png')


