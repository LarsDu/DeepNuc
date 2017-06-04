import tensorflow as tf
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pprint

import numpy as np
import os
import sys
import glob

import dubiotools as dbt
from onehotseqmutator import OnehotSeqMutator

sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from duseqlogo import LogoTools
import nucheatmap

import nucconvmodel
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
                 concat_revcom_input,
                 nn_method_key):

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

        self.nn_method_key = nn_method_key
        self.nn_method = nucconvmodel.methods_dict[nn_method_key]
        
        self.train_steps_per_epoch = int(self.train_batcher.num_records//self.batch_size)
        if self.test_batcher:
            self.test_steps_per_epoch = int(self.test_batcher.num_records//self.batch_size)
        self.num_steps = int(self.train_steps_per_epoch*self.num_epochs)
        

        self.save_on_epoch = 5 #This will be overrided in child class __init__
       
        self.train_metrics_vector = [] #a list of metrics recorded on each save_on_epoch
        self.test_metrics_vector =[]

        self.epoch = 0
        self.step=0

        #http://stackoverflow.com/questions/43218731/
        #Deprecated method of saving step on graph
        #self.global_step = tf.Variable(0, trainable=False,name='global_step')

        #Saver should be set in build_model() after all ops are declared
        self.saver = None

        
        
    def save(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        #Save checkpoint in tensorflow
        checkpoint_name = self.checkpoint_dir+os.sep+'checkpoints'
        self.saver.save(self.sess,checkpoint_name,global_step=self.step)

        #Save metrics using pickle in the metrics folder
        if not os.path.exists(self.metrics_dir):
            os.makedirs(self.metrics_dir)

        
        metrics_file = self.metrics_dir+os.sep+'metrics-'+str(self.step)+'.p'
        with open(metrics_file,'w') as of:
            pickle.dump(self.train_metrics_vector,of)
            pickle.dump(self.test_metrics_vector,of)

        #Clean the metrics directory of old pickle files (to save storage space)
        flist = glob.glob('*.p')
        flist_steps = [int(f.strip('.p').split('-')[1]) for f in flist]
        max_metric = max(flist_steps+[0])
        
        for f in flist:
            if max_metric != int(f.strip('.p').split('-')[1]):
                os.remove(f)

        
    def load(self,checkpoint_dir):
        '''
        Load saved model from checkpoint directory.
        '''
        print(" Retrieving checkpoints from", checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess,ckpt.model_checkpoint_path)
            print "\n\n\n\nSuccessfully loaded checkpoint from",ckpt.model_checkpoint_path
            #Extract step from checkpoint filename
            self.step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
            self.epoch = int(self.step//self.train_steps_per_epoch)

            self.load_pickle_metrics(self.step)

            return True
        else:
            print ("Failed to load checkpoint",checkpoint_dir)
            return False
        
    def load_pickle_metrics(self,step):
        #Load metrics from pickled metrics file
        metrics_file = self.metrics_dir+os.sep+'metrics-'+str(self.step)+'.p'
        with open(metrics_file,'r') as of:
            self.train_metrics_vector = pickle.load(of)
            if self.test_batcher:
                self.test_metrics_vector = pickle.load(of)
                print "Successfully loaded recorded metrics data from {}".\
                  format(metrics_file)

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

                       
        #If model already finished training, just return last metrics
        if self.step >= self.num_steps or self.epoch>self.num_epochs:
            print "Loaded model already finished training"
            print "Model was loaded at step {} epoch {} and num_steps set to {} and num epochs set to {}".format(self.step,self.epoch,self.num_steps,self.num_epochs)
   

            
        #Important note: epochs go from 1 to num_epochs inclusive. The
        # last epoch index is equal to num_epochs
        
            
        for _ in xrange(self.epoch,self.num_epochs):
            self.epoch += 1 
            for _ in xrange(self.train_steps_per_epoch):
                self.step += 1
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
                if (self.step % self.train_steps_per_epoch == 0):
                    # Print status to stdout.
                    print('Epoch %d Step %d loss = %.4f (%.3f sec)' % (self.epoch, self.step,
                                                     loss_value,
                                                      duration))

                    #Writer summary
                    summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)
                    self.summary_writer.add_summary(summary_str, self.step)
                    self.summary_writer.flush() #ensure summaries written to disk

                #Save checkpoint and evaluate training and test sets                     
                if ( self.epoch % self.save_on_epoch == 0
                     and self.epoch > 0
                     and self.epoch !=self.num_epochs
                     and self.step % self.train_steps_per_epoch == 0):
                    print('Training data eval:')
                    train_metrics=self.eval_model_metrics(self.train_batcher)
                    self.print_metrics(train_metrics)
                    self.train_metrics_vector.append(train_metrics)

                    if self.test_batcher != None:
                        print('Testing data eval:')
                        test_metrics=self.eval_model_metrics(self.test_batcher)
                        self.test_metrics_vector.append(test_metrics)
                        self.print_metrics(test_metrics)

                    print "Saving checkpoints"
                    #self.save()
                 
                if (self.epoch == self.num_epochs and self.step % self.train_steps_per_epoch ==0):
                    # This is the final step and epoch, save metrics
                    
                    # Evaluate the entire training set.
                    print('Training data eval:')
                    #self.eval_model_accuracy(self.train_batcher)
                    self.train_metrics_vector.append( self.eval_model_metrics(self.train_batcher,
                                                                 save_plots=True,
                                                                 image_name='train_metrics.png'))

                    if self.test_batcher != None:
                        print('Testing data eval:')
                        self.test_metrics_vector.append(self.eval_model_metrics(self.test_batcher,
                                                                    save_plots=True,
                                                                    image_name='test_metrics.png'))
                    print "Saving final checkpoint"
                    self.save()


        #Set return values 
        ret_metrics = []
        if self.train_metrics_vector != []:
            ret_metrics.append(self.train_metrics_vector[-1])
        else:
            ret_metrics.append([])
            
        if self.test_metrics_vector != []:
            ret_metrics.append(self.test_metrics_vector[-1])
        else:
            ret_metrics.append([])
      
        
        return ret_metrics


    def eval_batchers(self,save_plots=True):
        # Evaluate training and test batcher data.
        print('Training data eval:')
        #self.eval_model_accuracy(self.train_batcher)
        train_results_dict = self.eval_model_metrics(self.train_batcher,
                                                     save_plots=save_plots)

        self.print_metrics(train_results_dict)

        if self.test_batcher != None:
            print('Testing data eval:')
            test_results_dict = self.eval_model_metrics(self.test_batcher,
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



    def plot_test_epoch_vs_metric(self,
                                  metric_key="auroc",
                                  suffix = '',
                                  save_plot=True,
                                  xmin = 0.0,
                                  ymin=0.5):
        format_dict= {"auroc":"auROC","auPRC":"auprc","f1_score":"F1-Score"}
        num_mets = len(self.test_metrics_vector)
        if num_mets == 0:
            print "Test metrics vector is empty!"
            return None
        met_y = [m[metric_key] for m in self.test_metrics_vector]
        ep_x = [m["epoch"] for m in self.test_metrics_vector]
        
        fig,ax = plt.subplots(1)
        ax.plot(ep_x,met_y)

        ax.set_xlabel("Number of epochs")
        ax.set_xlim(xmin,ep_x[-1]+5)
        ax.set_ylim(ymin,1.0)
        if metric_key in format_dict:
            ax.set_title("Epoch vs. {} {}".format(format_dict[metric_key],suffix))
            ax.set_ylabel("{}".format(format_dict[metric_key]))
        else:
            ax.set_title("Epoch vs.{} {}".format(metric_key,suffix))
            ax.set_ylabel("{}".format(metric_key))
        if save_plot:
            plot_file = self.save_dir+os.sep+"epoch_vs_{}_{}.png".format(metric_key,suffix)
            fig.savefig(plot_file)

    


    ###Relevance batch methods###

    def relevance_from_nucs(self,nucs,label):
        """
        Return the relevance of a nucleotide sequence and corresponding label
                
        :nuc_seq: string nucleotide sequence
        :label: 1 x num_classes numpy indicator array
        :returns: 4xn relevance matrix
        :rtype: numpy array
        """

        return self.run_relevance(dbt.seq_to_onehot(nuc_seq),label)
        
    def run_relevance(self,onehot_seq,label):
        """
        Return the relevance of a onehot encoded sequence and corresponding label
                
        :onehot_seq: nx4 onehot representation of a nucleotide sequence
        :label: 1 x num_classes numpy indicator array
        :returns: 4xn relevance matrix
        :rtype: numpy array
        """
        
        feed_dict = {
            self.dna_seq_placeholder: np.expand_dims(onehot_seq,axis=0),
            self.labels_placeholder: np.expand_dims(label,axis=0),
            self.keep_prob_placeholder: 1.0
            }
        relevance_batch = self.sess.run(self.relevance,feed_dict=feed_dict)
        relevance = np.squeeze(relevance_batch[0],axis=0).T

        return relevance


    def relevance_from_batcher(self,batcher,index):
        """
        :param batcher: DataBatcher object
        :param index: integer index of item in DataBatcher object
        :returns: 4xn relevance matrix
        :rtype: numpy array
    
        """
        
        batch_size=1 #Needs to be 1 for now due to conv2d_transpose issue
        label, onehot_seq = batcher.pull_batch_by_index(index,batch_size)
        rel_mat = self.run_relevance(onehot_seq[0],label[0])
        return rel_mat

    def plot_relevance_logo_from_batcher(self,batcher,index):
        batch_size=1 #Needs to be 1 for now due to conv2d_transpose issue
        label, onehot_seq = batcher.pull_batch_by_index(index,batch_size)
        numeric_label = labels_batch[0].tolist().index(1)
        save_fig = self.save_dir+os.sep+'relevance_logo_ind{}_lab{}.png'.format(index,
                                                                        numeric_label)
        self.plot_relevance_logo(onehot_seq,label,save_fig)

        
    def plot_relevance_heatmap_from_batcher(self,batcher,index):
        batch_size=1 #Needs to be 1 for now due to conv2d_transpose issue
        labels_batch,dna_batch = batcher.pull_batch_by_index(index,batch_size)
        numeric_label = labels_batch[0].tolist().index(1)
        save_fig = self.save_dir+os.sep+'relevance_heat_ind{}_lab{}.png'.format(index,
                                                                        numeric_label)
        self.plot_relevance_heatmap(dna_batch[0],
                                    labels_batch[0],
                                    save_fig)
    

     

    def plot_relevance_heatmap(self,onehot_seq,label,save_fig):
        relevance = self.run_relevance(onehot_seq,label)
        seq = dbt.onehot_to_nuc(onehot_seq.T)
        
        fig,ax = nucheatmap.nuc_heatmap(seq,
                               relevance,
                               save_fig=save_fig,
                               clims = [0,np.max(relevance)],
                               cmap='Blues')
        

    
    def plot_relevance_logo(self,onehot_seq,label,save_fig):
        logosheets=[]
        input_seqs=[]
               
        np.set_printoptions(linewidth=500,precision=4)
        save_file = self.save_dir+save_fig

        
        relevance = self.relevance(onehot_seq,label)
        r_img = np.squeeze(relevance).T

       
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
        input_seqs.append(onehot_seq.T)

        rel_sheet = LogoTools.LogoNucSheet(logosheets,input_seqs,input_type='heights')
        rel_sheet.write_to_png(save_file)

        #plt.pcolor(r_img,cmap=plt.cm.Reds)
        #print "A relevance"
        #plt.plot(r_img[0,:])
        #print "Relevance by position"
        #plt.plot(np.sum(r_img,axis=0))


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





        
    ##Alipanahi mut map methods###

    def alipanahi_mutmap(self,onehot_seq,label):
        """
        Create an matrix representing the effects of every
        possible mutation on classification score as described in Alipanahi et al 2015

        :onehot_seq: nx4 onehot representation of a nucleotide sequence
        :label: 1 x num_classes numpy indicator array
        :returns: nx4 mutation map numpy array
        """
        
        #Mutate the pulled batch sequence.
        #OnehotSeqMutator will produce every SNP for the input sequence

        oh_iter = OnehotSeqMutator(onehot_seq.T) #4xn inputs

        
        eval_batch_size = 75 #Number of generated examples to process in parallel
                             # with each step

        single_pulls = oh_iter.n%eval_batch_size
        num_whole_batches = int(oh_iter.n//eval_batch_size+single_pulls)
        num_pulls = num_whole_batches+single_pulls
        
        all_probs = np.zeros((oh_iter.n,self.num_classes))

        
        for i in range(num_pulls):
            if i<num_whole_batches:
                iter_batch_size = eval_batch_size
            else:
                iter_batch_size=1
            
            labels_batch = np.asarray(iter_batch_size*[label])
            dna_seq_batch = oh_iter.pull_batch(iter_batch_size)
            feed_dict = {
                self.dna_seq_placeholder: dna_seq_batch,
                self.labels_placeholder: labels_batch,
                self.keep_prob_placeholder: 1.0
                }
            

            cur_probs = self.sess.run(self.probs,feed_dict=feed_dict)
            #TODO: Map these values back to the original nuc array
            

            if iter_batch_size > 1:
                start_ind = iter_batch_size*i
            elif iter_batch_size == 1:
                start_ind = num_whole_batches*eval_batch_size+(i-num_whole_batches)
            else:
                print "Never reach this condition"

            start_ind = iter_batch_size*i
            all_probs[start_ind:start_ind+iter_batch_size,:] = cur_probs

            
            
            #print "OHseqshape",onehot_seq.shape
            seq_len = onehot_seq.shape[0]
            amutmap_ds=np.zeros((seq_len,4))


            label_index = label.tolist().index(1)
            #Onehot seq mutator created SNPs in order
            #Fill output matrix with logits except where nucleotides unchanged
            #Remember onehot_seq is nx4 while nuc_heatmap takes inputs that are 4xn

            ps_feed_dict = {
                self.dna_seq_placeholder:np.expand_dims(onehot_seq,axis=0),
                self.labels_placeholder: np.expand_dims(label,axis=0),
                self.keep_prob_placeholder: 1.0
                }

            #ps is the original score of the original input sequence
            ps = self.sess.run(self.probs,feed_dict=ps_feed_dict)[0][label_index]
            k=0


            for i in range(seq_len):
                for j in range(4):
                    if onehot_seq[i,j] == 1:
                        amutmap_ds[i,j] = 0 #Set original letter to 0
                    else:
                        #ps_hat is the score for a given snp
                        ps_hat = all_probs[k,label_index]
                        amutmap_ds[i,j] = (ps_hat - ps)*max(0,ps_hat,ps)
                        k+=1

            #amutmap_ds is nx4
            
        return amutmap_ds


    def alipanahi_mutmap_from_batcher(self,batcher,index):
        label,onehot_seq = batcher.pull_batch_by_index(index,batch_size)
        return self.alipanahi_mutmap(onehot_seq,label)
    
    def plot_alipanahi_mutmap(self,onehot_seq,label,save_fig):
        seq = dbt.onehot_to_nuc(onehot_seq.T)
        amut_onehot = self.alipanahi_mutmap(onehot_seq,label)
        nucheatmap.nuc_heatmap(seq,amut_onehot.T,save_fig=save_fig)

    def plot_alipanahi_mutmap_from_batcher(self,batcher,index):
        batch_size = 1
        labels_batch, dna_seq_batch = batcher.pull_batch_by_index(index,batch_size)
        #print "Index {} has label {}".format(index,labels_batch[0])
        numeric_label = labels_batch[0].tolist().index(1)
        save_fig = self.save_dir+os.sep+'alipanahi_mut_map_ind{}_lab{}.png'.format(index,
                                                                                   numeric_label)
        self.plot_alipanahi_mutmap(dna_seq_batch[0],labels_batch[0],save_fig)
    

    def avg_alipanahi_mutmap_of_batcher(self,batcher):
        """Get every mutmap from a given batcher, then average over all
           mutation maps,
           Works for num_classes = 2"""

        all_labels, all_dna_seqs = batcher.pull_batch_by_index(0,batcher.num_records)

        amutmaps = [np.zeros((batcher.num_records,self.seq_len,4))]*self.num_classes
        
        for ci in range(self.num_classes):
            for ri in range(batcher.num_records):
                #double check this for errors
                #amutmap is nx4
                amutmaps[ci][ri,:,:] = self.alipanahi_mutmap(all_dna_seqs[ri,:,:],all_labels[ri,:])

        return [np.mean(amutmap,axis=0) for amutmap in amutmaps]

    def plot_avg_alipanahi_mutmap_of_batcher(self,batcher):

        amutmaps = self.avg_alipanahi_mutmap_of_batcher(batcher)
        
        for i,amap in enumerate(amutmaps):
            # Note: amax(arr, axis=1) give greatest val for each row (nuc for nx4)

            max_avg_nuc =(amap == np.amax(amap,axis=1,keepdims=True)).astype(np.float32)

            seq = dbt.onehot_to_nuc(max_avg_nuc.T)

            
            alipanahi_mutmap_dir = self.save_dir + os.sep+'alipanahi_mutmap_dir'
            if not os.path.exists(alipanahi_mutmap_dir):
                os.makedirs(alipanahi_mutmap_dir)

            save_fname = alipanahi_mutmap_dir+os.sep+'avg_batcher_mutmap_{}recs_class{}.png'.\
                                                        format(batcher.num_records,i)
            nucheatmap.nuc_heatmap(seq,amap.T,save_fig=save_fname)
        
    ###Mutmap methods###
    # generate every possible snp and measure change in logit

    
    

    
    def mutmap(self,onehot_seq,label):
        """
        :onehot_seq: nx4 onehot representation of a nucleotide sequence
        :label: integer numeric label (ie: 0 or 1 for a NucBinaryClassifier) 
        Create an matrix representing the effects of every
        possible mutation on classification score as described in Alipanahi et al 2015
        """

        #Mutate the pulled batch sequence.
        #OnehotSeqMutator will produce every SNP for the input sequence
        oh_iter = OnehotSeqMutator(onehot_seq.T) #4xn inputs

        
        eval_batch_size = 75 #Number of generated examples to process in parallel
                             # with each step

        single_pulls = oh_iter.n%eval_batch_size
        num_whole_batches = int(oh_iter.n//eval_batch_size+single_pulls)
        num_pulls = num_whole_batches+single_pulls
        
        all_logits = np.zeros((oh_iter.n,self.num_classes))

        
        for i in range(num_pulls):
            if i<num_whole_batches:
                iter_batch_size = eval_batch_size
            else:
                iter_batch_size=1
            
            labels_batch = np.asarray(iter_batch_size*[label])
            dna_seq_batch = oh_iter.pull_batch(iter_batch_size)
            feed_dict = {
                self.dna_seq_placeholder: dna_seq_batch,
                self.labels_placeholder: labels_batch,
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
            all_logits[start_ind:start_ind+iter_batch_size,:] = cur_logits

            #print "OHseqshape",onehot_seq.shape
            seq_len = onehot_seq.shape[0]
            mutmap_ds=np.zeros((seq_len,4))
            k=0
            
            label_index = label.tolist().index(1)
            #Onehot seq mutator created SNPs in order
            #Fill output matrix with logits except where nucleotides unchanged
            #Remember onehot_seq is nx4 while nuc_heatmap takes inputs that are 4xn
            for i in range(seq_len):
                for j in range(4):
                    if onehot_seq[i,j] == 1:
                        mutmap_ds[i,j] = 0 #Set original letter to 0
                    else:
                        mutmap_ds[i,j] = all_logits[k,label_index]
                        k+=1

        return mutmap_ds.T


    def mutmap_from_batcher(self,batcher,index):
        """
        Create an matrix representing the effects of every
        possible mutation on classification score as described in Alipanahi et al 2015.
        Retrieve this data from a databatcher
        """
        label, onehot_seq = batcher.pull_batch_by_index(index,batch_size=1)
        return self.mutmap(onehot_seq,label)

    
    def plot_mutmap(self,onehot_seq,label,save_fig):
        """
        :param onehot_seq: nx4 matrix
        :param label: 
        :returns: 
        :rtype: 
        """
        
        seq = dbt.onehot_to_nuc(onehot_seq.T)
        mut_onehot = self.mutmap(onehot_seq,label)
        #print "mut_onehot",mut_onehot.shape
        #print mut_onehot
        return nucheatmap.nuc_heatmap(seq,mut_onehot,save_fig=save_fig)

    def plot_mutmap_from_batcher(self,batcher,index):
        batch_size = 1
        labels_batch, dna_seq_batch = batcher.pull_batch_by_index(index,batch_size)
        #print "Index {} has label {}".format(index,labels_batch[0])
        numeric_label = labels_batch[0].tolist().index(1)
        save_fig = self.save_dir+os.sep+'mut_map_ind{}_lab{}.png'.format(index,numeric_label)
        return self.plot_mutmap(dna_seq_batch[0],labels_batch[0],save_fig)
    







    #################


    def print_global_variables(self):
        """Print all variable names from the current graph"""
        print "Printing global_variables"
        gvars =  list(tf.global_variables())
        for var in gvars:
            print "Variable name",var.name
            print self.sess.run(var)
            


    
                          
    def get_optimal_metrics(self,metrics_vector, metric_key="auroc"):
        """
        Get the metrics from the epoch where a given metric was at it maximum
        """
        best_val = 0
        for metric in metrics_vector:
            best_val = max(metric[metric_key],best_val)

        for metric in metrics_vector:
            #metric here is an OrderedDict of metrics
            if metric[metric_key]==best_val:
                return metric

