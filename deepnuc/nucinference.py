import tensorflow as tf
import numpy as np
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pprint

import numpy as np
import os
import sys

import dubiotools as dbt
from onehotseqmutator import OnehotSeqMutator

sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from duseqlogo import LogoTools
import nucheatmap

from collections import OrderedDict

import pickle

class NucInference(object):
    
    """
    Base class for NucBinaryClassifier and NucRegressor

    This class should contain all methods that work for both child classes.
    This includes train(),save(),and load(). Child classes must contain
    method eval_model_metrics()
    
    
    build_model() should be different due to different loss functions and lack
    of classification metrics.
    """
    
    use_onehot_labels = True

    def __init__(self,
                 sess,
                 train_batcher,
                 test_batcher,
                 num_epochs,
                 learning_rate,
                 batch_size,
                 seq_len,
                 save_dir,
                 keep_prob,
                 beta1,
                 concat_revcom_input):

        self.sess = sess
        self.train_batcher = train_batcher
        self.test_batcher = test_batcher
        self.seq_len = self.train_batcher.seq_len
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.save_dir = save_dir
        self.summary_dir = self.save_dir+os.sep+'summaries'
        self.checkpoint_dir = self.save_dir+os.sep+'checkpoints'
        self.metrics_dir = self.save_dir+os.sep+'metrics'
        #One minus the dropout_probability if dropout is enabled for a particular model
        self.keep_prob = 0.5
        #beta1 is a parameter for the AdamOptimizer
        self.beta1 = beta1

        #This flag will tell the inference method to concatenate
        #the reverse complemented version of the input sequence
        #to the input vector
        self.concat_revcom_input = concat_revcom_input
                
        self.train_steps_per_epoch = int(self.train_batcher.num_records//self.batch_size)
        self.test_steps_per_epoch = int(self.test_batcher.num_records//self.batch_size)
        self.total_iterations = int(self.train_steps_per_epoch*self.num_epochs)


        self.save_on_epoch = 5 #This will be overrided in child class __init__
        self.train_metrics_vector = [] #a list of metrics recorded on each save_on_epoch
        self.test_metrics_vector =[]

        self.epoch = 0
        self.step=0
        #http://stackoverflow.com/questions/43218731/
        self.global_step = tf.Variable(0, trainable=False,name='global_step')


        
    def save(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        #Save checkpoint in tensorflow
        checkpoint_name = self.checkpoint_dir+os.sep+'checkpoints'
        self.saver.save(self.sess,checkpoint_name,global_step=self.global_step)

        #Save metrics using pickle in the metrics folder
        if not os.path.exists(self.metrics_dir):
            os.makedirs(self.metrics_dir)

        metrics_file = self.metrics_dir+os.sep+'metrics-'+str(self.step+1)+'.p'
        with open(metrics_file,'w') as of:
            pickle.dump(self.train_metrics_vector,of)
            pickle.dump(self.test_metrics_vector,of)
        
        
    def load(self,checkpoint_dir):
        '''
        Load saved model from checkpoint directory.
        self.global_step is defined here
        '''
        print(" Retrieving checkpoints from", checkpoint_dir)
        print "Note: loading saved models won't restore previous model metrics"
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        
        if ckpt and ckpt.model_checkpoint_path:
            print ("Successfully loaded checkpoint from",checkpoint_dir)
            self.step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
            self.epoch = (self.step*self.batch_size)//self.train_batcher.num_records
            #Load metrics from pickled metrics file
            metrics_file = self.metrics_dir+os.sep+'metrics-'+str(self.step)+'.p'
            with open(metrics_file,'r') as of:
                print "Reading recorded metrics data from {}".format(metrics_file)
                self.train_metrics_vector = pickle.load(of)
                self.train_metrics_batcher = pickle.load(of)
            return True
        else:
            print ("Failed to load checkpoint",checkpoint_dir)
            return False
        


    def train(self):
        """
        Train a model
        :returns: Tuple of two dicts: training metrics, and testing metrics

        Note: This method was designed to work for both nucregressor and nucclassifier
        However, those objects should have different eval_model_metrics() methods, since
        classification and regression produce different metrics
        """
    
        #coord = tf.train.Coordinator()
        #threads = tf.train.start_queue_runners(self.sess.coord)

        
        
        start_time = time.time()
    
        train_results_dict={}
        test_results_dict={}
                   

        
        for epoch in xrange(self.num_epochs):

            for step in xrange(self.train_steps_per_epoch):

                (labels_batch,dna_seq_batch) = self.train_batcher.pull_batch(self.batch_size)
            
                feed_dict={
                    self.dna_seq_placeholder:dna_seq_batch,
                    self.labels_placeholder:labels_batch,
                    self.keep_prob_placeholder:self.keep_prob
                    }

                _,loss_value,_ =self.sess.run([self.train_op, self.loss, self.logits],
                    feed_dict=feed_dict)

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                duration = time.time() - start_time
                
                # Write the summaries and print an overview fairly often.
                if epoch % 1 == 0 and epoch > 0 and (step % self.train_steps_per_epoch == 0):
                    # Print status to stdout.
                    print('Epoch %d Step %d loss = %.4f (%.3f sec)' % (self.epoch, self.step,
                                                     loss_value,
                                                      duration))

                    #Writer summary
                    summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)
                    self.summary_writer.add_summary(summary_str, self.step)
                    self.summary_writer.flush() #ensure summaries written to disk

                #Save checkpoint and evaluate training and test sets                     
                if ( epoch % self.save_on_epoch == 0
                     and epoch>0 and (step % self.train_steps_per_epoch == 0)
                                    and ((step + 1) != self.total_iterations) ):
                    print "Saving checkpoints"
                    self.save()
                    print('Training data eval:')
                    train_metrics=self.eval_model_metrics(self.train_batcher)
                    self.print_metrics(train_metrics)
                    self.train_metrics_vector.append(train_metrics)

                    if self.test_batcher != None:
                        print('Testing data eval:')
                        test_metrics=self.eval_model_metrics(self.test_batcher)
                        self.test_metrics_vector.append(test_metrics)
                        self.print_metrics(test_metrics)
                if ((step + 1) == self.total_iterations):
                    # For the last iteration, save metrics
                    print "Saving final checkpoint"
                    self.save(self.checkpoint_dir)

                    # Evaluate the entire training set.
                    print('Training data eval:')
                    #self.eval_model_accuracy(self.train_batcher)
                    train_results_dict = self.eval_model_metrics(self.train_batcher,
                                                                 show_plots=False,
                                                                 save_plots=True)

                    if self.test_batcher != None:
                        print('Testing data eval:')
                        test_results_dict = self.eval_model_metrics(self.test_batcher,
                                                                    show_plots=False,
                                                                    save_plots=True)

                self.step += 1
            self.epoch += 1
        
        return (train_results_dict,test_results_dict)


    def eval_train_test(self,show_plots=False,save_plots=True):
        # Evaluate the entire training set.
        print('Training data eval:')
        #self.eval_model_accuracy(self.train_batcher)
        train_results_dict = self.eval_model_metrics(self.train_batcher,
                                                     show_plots=show_plots,
                                                     save_plots=save_plots)

        self.print_metrics(train_results_dict)

        if self.test_batcher != None:
            print('Testing data eval:')
            test_results_dict = self.eval_model_metrics(self.test_batcher,
                                                        show_plots=show_plots,
                                                        save_plots=save_plots)
        self.print_metrics(test_results_dict)
    
    def print_metrics(self,metrics_dict):
        for key,value in metrics_dict.viewitems():
            #Do not print out arrays!
            if type(value) != np.ndarray:
                print '\t',key,":\t",value
            
            
    def eval_batch(self,dna_seq_batch,labels_batch):
        """ Evaluate a single batch of labels and data """
        feed_dict = {
            self.dna_seq_placeholder: dna_seq_batch,
            self.labels_placeholder: labels_batch,
            self.keep_prob_placeholder: 1.0
            }
        batch_logits,batch_network = self.sess.run(self.nn_method,feed_dict=feed_dict)
                                         
        return batch_logits,batch_network



    def get_optimal_metrics(self,metrics_vector, metric_key="auroc"):
        """
        Get the epoch value for the 
        """
        best_val = 0
        for metric in metrics_vector:
            best_val = max(metric[metric_key],best_val)

        for metric in metrics_vector:
            #metric here is an OrderedDict of metrics
            if metric[metric_key]==best_val:
                return metric
    
    def relevance_batch(self,dna_seq_batch,labels_batch):
        """ Return the relevance of a single batch of labels and data """
        feed_dict = {
            self.dna_seq_placeholder: dna_seq_batch,
            self.labels_placeholder: labels_batch,
            self.keep_prob_placeholder: 1.0
            }
        batch_relevance = self.sess.run(self.relevance,feed_dict=feed_dict)
        return batch_relevance


    def relevance_batch_by_index(self,batcher,index):
        batch_size=1 #Needs to be 1 for now due to conv2d_transpose issue
        labels_batch, dna_seq_batch = batcher.pull_batch_by_index(index,batch_size)
        rel_mat = self.relevance_batch_plot(labels_batch,dna_seq_batch,"decomp_"+str(index)+'.png')
        return rel_mat


    def mutation_map_ds_from_batcher(self,batcher,index):
        """
        Create an matrix representing the effects of every
        possible mutation on classification score as described in Alipanahi et al 2015.
        Retrieve this data from a databatcher
        """
        label, onehot_seq = batcher.pull_batch_by_index(index,batch_size=1)
        return self.mutation_map_ds(label,onehot_seq)


    
    def mutation_map_ds(self,onehot_seq,label = 1):
        """
        Create an matrix representing the effects of every
        possible mutation on classification score as described in Alipanahi et al 2015
        """

        #Mutate the pulled batch sequence.
        #OnehotSeqMutator will produce every SNP for the input sequence
        oh_iter = (OnehotSeqMutator(onehot_seq))

        
        eval_batch_size = 75 #Number of generated examples to process in parallel
                             # with each step

        single_pulls = oh_iter.n%eval_batch_size
        num_whole_batches = int(oh_iter.n-single_pulls)
        num_pulls = num_whole_batches+single_pulls

        all_logits = np.zeros((oh_iter.n))
                               
        for i in range(num_pulls):
            if i<num_whole_batches:
                iter_batch_size = eval_batch_size
            else:
                iter_batch_size=1
            dna_seq_batch = oh_iter.pull_batch(iter_batch_size)
            feed_dict = {
                self.dna_seq_placeholder: dna_seq_batch,
                self.labels_placeholder: label,
                self.keep_prob_placeholder: 1.0
                }
            
            
            cur_logits = self.sess.run(self.logits,feed_dict=feed_dict)
            #TODO: Map these values back to the original nuc array


            if iter_batch_size > 1:
                start_ind = iter_batch_size*i
            elif iter_batch_size == 1:
                start_ind = num_whole_batches*eval_batch_size+(i-num_whole_batches)
            else:
                print "Never reach this condition"

            start_ind = iter_batch_size*i
            all_logits[start_ind:start_ind+iter_batch_size] = cur_logits


            mutmap_ds=np.zeros_like(onehot_seq)
            k=0
            #Onehot seq mutator created SNPs in order
            #Fill output matrix with logits except where nucleotides unchanged
            for i in range(4):
                for j in range(onehot_seq.shape[1]):
                    if onehot_seq[i,j] == 1:
                        mutmap_ds[i,j] = 0 #Set original letter to 0
                    else:
                        mutmap_ds[i,j] = all_logits[k]
                        k+=1
            
        return mutmap_ds


    def mutation_map_ds_heatmap(self,onehot_seq,label):
        #onehot_seq = dbt.seq_to_onehot(seq)
        mut_onehot = self.mutation_map_ds(onehot_seq,label)
        seq = dbt.onehot_to_seq(onehot_seq)
        nucheatmap.nuc_heatmap(seq,mut_onehot)

    def mutation_map_ds_heatmap_batcher(self,batcher,index):
        batch_size = 1
        labels_batch, dna_seq_batch = batcher.pull_batch_by_index(index,batch_size)
        self.mutation_map_ds_heatmap(self,dna_seq_batch[0],labels_batch[0])
    
    def relevance_batch_plot(self,labels_batch,dna_seq_batch,image_name='relevance.png'):
        logosheets=[]
        input_seqs=[]

        feed_dict={
                   self.dna_seq_placeholder:dna_seq_batch,
                   self.labels_placeholder:labels_batch,
                   self.keep_prob_placeholder:1.0
                  }

        #Note that this should only hold if batch_size=1
        #flat_relevance = tf.reshape(relevance,[-1])
        r_input = self.sess.run(self.relevance,
                                feed_dict=feed_dict)
        r_img = np.squeeze(r_input).T

        #print r_img
        #print r_img.shape

        np.set_printoptions(linewidth=500,precision=4)

        
        ###Build a "relevance scaled position weight matrix"
        #Convert each position to a position probability matrix
        r_ppm = r_img/np.sum(r_img,axis=0)
        
        lh = LogoTools.PwmTools.ppm_to_logo_heights(r_ppm)
        #Relevance scale logo_heights
        r_rel =np.sum(r_img,axis=0) #relavance by position
        max_relevance = np.max(r_rel)
        min_relevance = np.min(r_rel)
        #print "r_rel max", max_relevance
        #print "r_rel min", min_relevance

        #lh is in bits of information
        #Rescale logo_heights to r_rel
        scaled_lh = lh * r_rel/(max_relevance - min_relevance)
        logosheets.append(scaled_lh*25)
        input_seqs.append(dna_seq_batch[0].T)

        save_file = self.save_dir+image_name
        rel_sheet = LogoTools.LogoNucSheet(logosheets,input_seqs,input_type='heights')
        rel_sheet.write_to_png(save_file)

        #Plot heatmap

        heatmap_im = os.path.splitext(image_name)[0]+'_heatmap.png'
        seq = dbt.onehot_to_nuc(dna_seq_batch[0].T)
        print labels_batch
        nucheatmap.nuc_heatmap(seq,r_img,heatmap_im,heatmap_im)










        
        #plt.pcolor(r_img,cmap=plt.cm.Reds)
        #plt.show()

        #print "A relevance"
        #plt.plot(r_img[0,:])
        #plt.show()
        #print "Relevance by position"
        #plt.plot(np.sum(r_img,axis=0))
        #plt.show()


        #logits_np = self.sess.run(self.logits,
        #                 feed_dict=feed_dict)


        #Print actual label and inference if classification 
        #guess = logits_np.tolist()
        #guess = guess[0].index(max(guess[0]))
        #actual = labels_batch[0].tolist().index(1.)
        #print logits_np
        #print self.sess.run(self.probs,feed_dict=feed_dict)
        #print ("Guess:",(guess))
        #print ("Actual:",(actual))


