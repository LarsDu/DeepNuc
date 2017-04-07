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

from onehotseqmutator import OnehotSeqMutator

sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from duseqlogo import LogoTools


class NucInference:
    
    """
    Base class for NucBinaryClassifier and NucRegressor

    This class should contain all methods that work for both child classes.
    This includes train(),save(),and load(). Child classes must contain
    method eval_model_metrics()
    
    
    build_model() should be different due to different loss functions and lack
    of classification metrics.
    """
    
    use_onehot_labels = True

    
    def save(self,checkpoint_dir,step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_name = checkpoint_dir+os.sep+'checkpoints'
        self.saver.save(self.sess,
                        checkpoint_name,
                        global_step = step)

    def load(self,checkpoint_dir):
        print(" Retrieving checkpoints from", checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print ("Successfully loaded checkpoint from",checkpoint_dir)
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
        step = 0
    
        train_results_dict={}
        test_results_dict={}
                   
        
        for epoch in xrange(self.num_epochs):
            for i in xrange(self.train_steps_per_epoch):
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
                    print('Epoch %d Step %d loss = %.4f (%.3f sec)' % (epoch, step,
                                                     loss_value,
                                                      duration))

                    #Writer summary
                    summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)
                    self.summary_writer.add_summary(summary_str, step)
                    self.summary_writer.flush() #ensure summaries written to disk

                #Save checkpoint and evaluate training and test sets                     
                if ( epoch % 10 == 0 and epoch>0 and (step % self.train_steps_per_epoch == 0)
                                    and ((step + 1) != self.total_iterations) ):
                    print "Saving checkpoints"
                    self.save(self.checkpoint_dir, step)
                    print('Training data eval:')
                    self.eval_model_metrics(self.train_batcher)
                    if self.test_batcher != None:
                        print('Testing data eval:')
                        self.eval_model_metrics(self.test_batcher)
                        
                if ((step + 1) == self.total_iterations):
                    # For the last iteration, save metrics
                    print "Saving final checkpoint"
                    self.save(self.checkpoint_dir, step)

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

                step += 1
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
            if type(value) != np.ndarray:
                print key,":\t",value
            
            
    def eval_batch(self,dna_seq_batch,labels_batch):
        """ Evaluate a single batch of labels and data """
        feed_dict = {
            self.dna_seq_placeholder: dna_seq_batch,
            self.labels_placeholder: labels_batch,
            self.keep_prob_placeholder: 1.0
            }
        batch_logits,batch_network = self.sess.run(self.nn_method,feed_dict=feed_dict)
                                         
        return batch_logits,batch_network


    
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


    def mutation_map_delta_s_by_index(self,batcher,index):
        """Create an matrix representing the effects of every
        possible mutation on classification score as described in Alipanahi et al 2015
        
        :param batcher: DataBatcher to pull examples from 
        :param index: Index of the label and nucleotide sequence to be pulled
        :returns: 
        :rtype: 

        """
        label, onehot_seq = batcher.pull_batch_by_index(index,batch_size=1)

        #Mutate the pulled batch sequence.
        #Convert 
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

            start_ind = iter_batch_size*i
            all_logits[start_ind:start_ind+batch_size] = cur_logits[0]

        
        return all_logits


    
            
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

        print r_img
        print r_img.shape

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


