import numpy as np
import json
import os
import time

#from databatcher import DataBatcher
import jsonparams
import nucconvmodel
#import dubiotools as dbt


import tensorflow as tf
import sklearn.metrics as metrics

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pprint
from itertools import cycle

import sys
import os.path
sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from duseqlogo import LogoTools


#Logging imports
from logger import Logger
import sys

class NucClassifier:
    def __init__(self,
                 sess,
                 train_batcher,
                 test_batcher,
                 num_epochs,
                 learning_rate,
                 batch_size,
                 seq_len,
                 save_dir,
                 keep_prob=0.5,
                 beta1=0.9):
        
    
    
        self.sess = sess
        self.train_batcher = train_batcher
        self.test_batcher = test_batcher
        self.num_classes = self.train_batcher.num_classes
        self.seq_len = self.train_batcher.seq_len

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.save_dir = save_dir
        self.summary_dir = self.save_dir+os.sep+'summaries'
        self.checkpoint_dir = self.save_dir+os.sep+'checkpoints'

        #One minus the dropout_probability if dropout is enabled for a particular model
        self.keep_prob = 0.5
        #beta1 is a parameter for the AdamOptimizer
        self.beta1 = beta1

        
        self.train_steps_per_epoch = int(self.train_batcher.num_records//self.batch_size)
        self.test_steps_per_epoch = int(self.test_batcher.num_records//self.batch_size)
        self.total_iterations = int(self.train_steps_per_epoch*self.num_epochs)

            

    def build_model(self,nn_method):

        self.nn_method = nn_method

        self.dna_seq_placeholder = tf.placeholder(tf.float32,
                                          shape=[None,4,self.seq_len],
                                          name="dna_seq")

        self.labels_placeholder = tf.placeholder(tf.float32,
                                            shape=[None, self.num_classes],
                                            name="labels")

        self.keep_prob_placeholder = tf.placeholder(tf.float32,name="keep_prob")

        self.logits, self.network = self.nn_method(self.dna_seq_placeholder,
                                                   self.keep_prob_placeholder,
                                                   self.num_classes)

        if self.num_classes = 2:
            self.probs = tf.nn.sigmoid(self.logits)
        else:
            self.probs = tf.nn.softmax(self.logits)

                    
        self.loss = tf.reduce_mean(
                           tf.nn.sigmoid_cross_entropy_with_logits(self.logits,
                                                                   self.labels_placeholder))


        # Add gradient ops to graph with learning rate
        self.train_op = tf.train.AdamOptimizer(self.learning_rate,beta1=self.beta1).\
                                                         minimize(self.loss)

        '''
        Calculate metrics. num_true positives is the number of true positives for the current batch

        Table below shows index if tf.argmax is applied
       
        +-----+-----------+---------+
        |     | Classifier|  Label  |  
        +-----+-----------+---------+
        | TP  |     1     |  1      |  
        +-----+-----------+---------+
        | FP  |     1     |  0      |
        +-----+-----------+---------+
        | TN  |     0     |  0      |
        +-----+-----------+---------+
        | FN  |     0     |  1      |
        +-----+-----------+---------+

        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN) 
        F1-score = 2*(Prec*Rec)/(Prec+Rec)

        # Note: I ended up not using the tp,fp,tn,fn ops because I ended up calculating
        # these metrics using sklearn.
        '''

        #correct  = TN+TP     #Used for calculating accuracy
        self.logits_ind = tf.argmax(self.logits,1)
        self.labels_ind = tf.argmax(self.labels_placeholder,1)

        #Create max_mask of logits (ie: [-.5,.5] --> [0 1]. Note logits have
        # shape [batch_size * num_classes= 2]
        #self.inverse_logits_col = tf.ones_like(self.logits_ind) - self.logits_ind
        #self.max_mask_logits = tf.concat([self.inverse_logits_col,self.logits_ind],1)
       
        #True positives where logits_ind+labels_ind == 2
        #True negatives where logits_ind+labels_ind == 0 
        self.sum_ind = tf.add(self.logits_ind,self.labels_ind) 
        
        self.true_positives = tf.equal(self.sum_ind,2*tf.ones_like(self.sum_ind)) #bool
        self.num_true_positives =tf.reduce_sum(tf.cast(self.true_positives, tf.int32))

        #For FP classifier index > label index
        self.false_positives=tf.greater(self.logits_ind,self.labels_ind)
        self.num_false_positives = tf.reduce_sum(tf.cast(self.false_positives, tf.int32))
        
        self.true_negatives = tf.equal(self.sum_ind,tf.zeros_like(self.sum_ind)) #bool
        self.num_true_negatives= tf.reduce_sum(tf.cast(self.true_negatives,tf.int32))

        #For FN classifier index < label index
        self.false_negatives=tf.less(self.logits_ind,self.labels_ind)
        self.num_false_negatives = tf.reduce_sum(tf.cast(self.false_negatives,tf.int32))

        #num correct is also equal to TP+TN and can be used to calculate accuracy
        self.correct = tf.equal(self.logits_ind,self.labels_ind)
        self.num_correct= tf.reduce_sum(tf.cast(self.correct, tf.int32))
        
        
        
               
        self.relevance =self.network.relevance_backprop(tf.mul(self.logits,
                                                               self.labels_placeholder))


        '''Write and consolidate summaries'''
        self.loss_summary = tf.summary.scalar('loss',self.loss)

        self.summary_writer = tf.summary.FileWriter(self.summary_dir,self.sess.graph)
        
        
        self.summary_op = tf.summary.merge([self.loss_summary])
        
        self.vars = tf.trainable_variables()
        self.var_names = [var.name for var in self.vars] 
        #print "Trainable variables:\n"
        #for vname in self.var_names:
        #    print vname
            
        self.saver = tf.train.Saver(self.vars)
        
        #Note: Do not use tf.summary.merge_all() here. This will break encapsulation for
        # cross validation and lead to crashes when training multiple models
        
        tf.global_variables_initializer().run() 

        if self.load(self.checkpoint_dir):
            print "Successfully loaded from checkpoint",self.checkpoint_dir

        

                    
    def train(self):
        """
        Train a model

        :returns: Tuple of two dicts: training metrics, and testing metrics
        """
    
        #coord = tf.train.Coordinator()
        #threads = tf.train.start_queue_runners(self.sess.coord)

        start_time = time.time()
        step = 0
    
        train_results_dict={}
        test_results_dict={}
                   
        
        for epoch in xrange(self.params.num_epochs):
            for i in xrange(self.train_steps_per_epoch):
                (labels_batch,dna_seq_batch) = self.train_batcher.pull_batch(self.batch_size)
            
                feed_dict={
                    self.dna_seq_placeholder:dna_seq_batch,
                    self.labels_placeholder:labels_batch,
                    self.keep_prob_placeholder:self.params.keep_prob
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
                if ( epoch % 5 == 0 and epoch>0 and (step % self.train_steps_per_epoch == 0)
                                    and ((step + 1) != self.total_iterations) ):
                    print "Saving checkpoints"
                    self.save(self.checkpoint_dir, step)
                    print('Training data eval:')
                    self.eval_model_accuracy(self.train_batcher)
                    if self.test_batcher != None:
                        print('Testing data eval:')
                        self.eval_model_accuracy(self.test_batcher)
                        
                if ((step + 1) == self.total_iterations):
                    # For the last iteration, save metrics
                    print "Saving final checkpoint"
                    self.save(self.checkpoint_dir, step)

                    # Evaluate the entire training set.
                    print('Training data eval:')
                    #self.eval_model_accuracy(self.train_batcher)
                    train_results_dict = self.eval_model_metrics(self.train_batcher,
                                                                 pindex=1,
                                                                 show_plots=False,
                                                                 save_plots=True)
                    if self.test_batcher != None:
                        print('Testing data eval:')
                        test_results_dict = self.eval_model_metrics(self.test_batcher,
                                                                    pindex=1,
                                                                    show_plots=False,
                                                                    save_plots=True)

                step += 1
        return (train_results_dict,test_results_dict)




                        
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


    def relevance_batch_by_index(self,batcher,index,batch_size=1):
        labels_batch, dna_seq_batch = batcher.pull_batch_by_index(index,batch_size)
        rel_mat = self.relevance_batch_plot(labels_batch,dna_seq_batch)
        return rel_mat

        
    def relevance_batch_plot(self,labels_batch,dna_seq_batch):
        
        #image_dir = save_dir+os.sep+"img"
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

        r_img = np.transpose(np.squeeze(r_input[:,:,:,:],axis=(0,1)))

            
        np.set_printoptions(linewidth=500,precision=4)

        plt.pcolor(r_img,cmap=plt.cm.Reds)
        #plt.show()

        #print "A relevance"
        #plt.plot(r_img[0,:])
        #plt.show()
        #print "Relevance by position"
        #plt.plot(np.sum(r_img,axis=0))
        #plt.show()


        logits_np = self.sess.run(self.logits,
                         feed_dict=feed_dict)

        guess = logits_np.tolist()
        guess = guess[0].index(max(guess[0]))

        actual = labels_batch[0].tolist().index(1.)
    
        #print logits_np
        print self.sess.run(self.probs,feed_dict=feed_dict)
        print ("Guess:",(guess))
        print ("Actual:",(actual))

        ###Build a "relevance scaled position weight matrix"
        #Convert each position to a position probability matrix
        r_ppm = r_img/np.sum(r_img,axis=0)
        lh = LogoTools.PwmTools.ppm_to_logo_heights(r_ppm)
        #Relevance scale logo_heights
        r_rel =np.sum(r_img,axis=0) #relavance by position
        max_relevance = np.max(r_rel)
        min_relevance = np.min(r_rel)
        print "r_rel max", max_relevance
        print "r_rel min", min_relevance
        #lh is in bits of information
        #Rescale logo_heights to r_rel
        scaled_lh = lh * r_rel/(max_relevance - min_relevance)

        logosheets.append(scaled_lh*25)
        input_seqs.append(dna_seq_batch[0])

        save_file = self.save_dir+'relevance_query.png'
        rel_sheet = LogoTools.LogoNucSheet(logosheets,input_seqs,input_type='heights')
        rel_sheet.write_to_png(save_file)

        


    
    def eval_model_accuracy(self,batcher,eval_batch_size=50):
        
        num_correct = 0 #counts number of correct predictions
        batcher_steps_per_epoch =  batcher.num_records//eval_batch_size
        left_over_steps = batcher.num_records%eval_batch_size
        num_steps = batcher_steps_per_epoch+left_over_steps
        for i in range(num_steps):
            if i >= batcher_steps_per_epoch-1:
                batch_size=1
            else:
                batch_size=eval_batch_size

            labels_batch,dna_seq_batch = batcher.pull_batch(batch_size)
            feed_dict = {
                          self.dna_seq_placeholder:dna_seq_batch,
                          self.labels_placeholder:labels_batch,
                          self.keep_prob_placeholder:1.0
                        }
            ncorr,cur_prob = self.sess.run([self.num_correct,self.probs],
                                                  feed_dict=feed_dict)
            num_correct += ncorr

        accuracy = num_correct/float(batcher.num_records)
        print 'Num examples: %d  Num correct: %d  Accuracy: %0.04f' % \
                  (batcher.num_records, num_correct, accuracy)+'\n'
        return accuracy
            


            

    def eval_model_metrics(self,
                           batcher,
                           pindex=1,
                           show_plots=False,
                           save_plots=False,
                           eval_batch_size=50):

        """
        Note: This method only works for binary classification at the moment
        as auPRC and auROC graphs only apply to binary classificaton problems.
    
        TODO: Modify this code to perform auROC generation
        for one-vs-all in the case of multiclass classification.

        """
                
        #Ref: http://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics
        ##auROC calculations

        #Keep batch size at 1 for now to ensure 1 full epoch is evaluated
    

        all_labels = np.zeros((batcher.num_records,self.num_classes), dtype = np.float32)
        all_probs = np.zeros((batcher.num_records,self.num_classes), dtype = np.float32)
       
        num_correct = 0 #counts number of correct predictions
        batcher_steps_per_epoch =  batcher.num_records//eval_batch_size
        left_over_steps = batcher.num_records%eval_batch_size
        num_steps = batcher_steps_per_epoch+left_over_steps

        for i in range(num_steps):
            if i >= batcher_steps_per_epoch-1:
                batch_size=1
            else:
                batch_size=eval_batch_size

            labels_batch, dna_seq_batch = batcher.pull_batch(batch_size)
            feed_dict = {
                          self.dna_seq_placeholder:dna_seq_batch,
                          self.labels_placeholder:labels_batch,
                          self.keep_prob_placeholder:1.0
                        }

                
            ncorr,cur_prob= self.sess.run([self.num_correct,self.probs],feed_dict=feed_dict)
            num_correct += ncorr
            
            #Fill labels array
            all_labels[i:i+batch_size,:] =  labels_batch[0]
            all_probs[i:i+batch_size,:]  = cur_prob[0]


        #Calculate metrics and save results in a dict
        md = self.calculate_metrics(all_labels,all_probs,pindex)
    

        #########TODO
        ###Plot some metrics
        plot_colors = cycle(['cyan','blue','orange','teal'])
        
        #print "Labels shape",all_labels.shape
        #print "Probs shape",all_probs.shape
        #print "Preds shape",all_preds.shape

        #Generate auROC plot axes
        fig1,ax1  = plt.subplots(2)
        fig1.subplots_adjust(bottom=0.2)

        ax1[0].plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
        ax1[0].set_xbound(0.0,1.0)
        ax1[0].set_ybound(0.0,1.05)
        ax1[0].set_xlabel('False Positive Rate')
        ax1[0].set_ylabel('True Positive Rate')
        ax1[0].set_title('auROC')
        #plt.legend(loc='lower right')


        #Generate auPRC plot axes
        #ax1[1].plot([0,0.5],[1,0.5],color='royalblue',lw=2,linestyle='--')
        ax1[1].set_xbound(0.0,1.0)
        ax1[1].set_ybound(0.0,1.05)
        ax1[1].set_xlabel('Precision')
        ax1[1].set_ylabel('Recall')
        ax1[1].set_title('auPRC')

        ax1[0].plot(md["fpr"],md["tpr"],color=plot_colors.next(),
                      lw=2,linestyle='-',label='auROC curve (area=%0.2f)' % md["auroc"] )

        ax1[1].plot(md["thresh_precision"],md["thresh_recall"],color=plot_colors.next(),
                    lw=2,linestyle='-',label='auPRC curve (area=%0.2f)' % md["auprc"] )


        #print "auROC for class", pindex, "is", md["auroc"]
        #print "auPRC for class", pindex, "is", md["auprc"]

        #Note: avg prec score is the area under the prec recall curve

        #Note: Presumably class 1 (pos examples) should be the only f1 score we focus on
        #print "F1 score for class",i,"is",f1_score


        accuracy = float(num_correct)/batcher.num_records
        md["num_correct"] = num_correct
        md["accuracy"] = accuracy
        print 'Num examples: %d  Num correct: %d  Accuracy: %0.04f' % \
                  (batcher.num_records, num_correct, accuracy)+'\n'

        if show_plots:
            plt.tight_layout()
            plt.show()
        if save_plots:
            plt.tight_layout()
            plt_fname = self.save_dir+os.sep+'metrics.png'
            print "Saving auROC image to",plt_fname
            fig1.savefig(plt_fname)

        #Return metrics dictionary
        return md


    
    def calculate_metrics(self,all_labels,all_probs,pindex):
        """Calculate some metrics for the dataset
           return dictionary with metrics

        :param all_probs: nx2 prob values
        :param all_labels: nx2 labels
        :param pindex:  Positive class index
        :returns: dictionary of metrics
        :rtype: dict()

        """
        num_records = all_probs.shape[0]
        all_preds = np.zeros((num_records, self.num_classes),dtype = np.float32)
        all_preds[np.arange(num_records),all_probs.argmax(1)] = 1


        ###Calculate auROC
        #http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        #metrics.roc_curve(y_true, y_score[, ...]) #y_score is probs

        fpr,tpr,_ = metrics.roc_curve(all_labels[:,pindex],all_probs[:,pindex],pos_label=1)
        auroc = metrics.auc(fpr,tpr)
        
        thresh_precision,thresh_recall,prc_thresholds = metrics.precision_recall_curve(
                                                        all_labels[:,pindex],all_probs[:,pindex])
        
        #Calculate precision, recall, and f1-score for threshold = 0.5
        #confusion_matrix = metrics.confusion_matrix(all_labels[:,pindex],all_probs[:,pindex])
        precision, recall, f1_score, support = metrics.precision_recall_fscore_support(
                                                all_labels[:,pindex],
                                                all_preds[:,pindex],
                                                pos_label=pindex)

        precision = precision[pindex]
        recall = recall[pindex]
        f1_score = f1_score[pindex]
        support = support[pindex]
        
        auprc = metrics.average_precision_score(all_labels[:,pindex],all_probs[:,pindex])
        
        return {
                "auroc":auroc,
                "auprc":auprc,
                "fpr":fpr,
                "tpr":tpr,
                "precision":precision,
                "recall":recall,
                "f1_score":f1_score,
                "support":support,
                "thresh_precision":thresh_precision,
                "thresh_recall":thresh_recall,
                "prc_thresholds":prc_thresholds
                 }



            
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


    
      
    
