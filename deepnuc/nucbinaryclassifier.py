import tensorflow as tf
import numpy as np
import sklearn.metrics as metrics

#from databatcher import DataBatcher
import nucconvmodel
#import dubiotools as dbt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pprint
from itertools import cycle

import os
import sys


#Logging imports
from logger import Logger

from nucinference import NucInference

from collections import OrderedDict

class NucBinaryClassifier(NucInference):

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
                 keep_prob=0.5,
                 beta1=0.9,
                 concat_revcom_input=False,
                 nn_method_key="inferenceA",
                 pos_index=1):


        super(NucBinaryClassifier, self).__init__(sess,
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
                                    nn_method_key="inferenceA")


        if self.train_batcher.num_classes != 2:
            print "Error, more than two classes detected in train batcher"
        else:
            self.num_classes = 2

        #The index for the label that should be considered the positive class
        self.pos_index=pos_index
        self.save_on_epoch = 5


    def build_model(self):

        
        self.dna_seq_placeholder = tf.placeholder(tf.float32,
                                          shape=[None,self.seq_len,4],
                                          name="dna_seq")

        self.labels_placeholder = tf.placeholder(tf.float32,
                                            shape=[None, self.num_classes],
                                            name="labels")

        self.keep_prob_placeholder = tf.placeholder(tf.float32,name="keep_prob")

        self.logits, self.network = self.nn_method(self.dna_seq_placeholder,
                                                   self.keep_prob_placeholder,
                                                   self.num_classes)

        self.probs = tf.nn.softmax(self.logits)


        
        self.loss = tf.reduce_mean(
                           tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_placeholder,
                                                                   logits=self.logits))

        
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

        #num correct can be used to calculate accuracy
        self.correct = tf.equal(self.logits_ind,self.labels_ind)
        self.num_correct= tf.reduce_sum(tf.cast(self.correct, tf.int32))
                
               
        self.relevance =self.network.relevance_backprop(tf.multiply(self.logits,
                                                               self.labels_placeholder))
    
        
        '''Write and consolidate summaries'''
        self.loss_summary = tf.summary.scalar('loss',self.loss)

        self.summary_writer = tf.summary.FileWriter(self.summary_dir,self.sess.graph)
        
        
        self.summary_op = tf.summary.merge([self.loss_summary])
        
        
        #Note: Do not use tf.summary.merge_all() here. This will break encapsulation for
        # cross validation and lead to crashes when training multiple models
        
        # Add gradient ops to graph with learning rate
        self.train_op = tf.train.AdamOptimizer(self.learning_rate,
                                               beta1=self.beta1).minimize(self.loss)


        
        self.vars = tf.trainable_variables()
        self.var_names = [var.name for var in self.vars] 
        #print "Trainable variables:\n"
        #for vname in self.var_names:
        #    print vname
            
        self.saver = tf.train.Saver()
        self.init_op = tf.global_variables_initializer()
        #Important note: Restoring model does not require init_op.
        #In fact calling tf.global_variables_initializer() after loading a model
        #will overwrite loaded weights
        self.sess.run(self.init_op)
        self.load(self.checkpoint_dir)
        
    def eval_model_metrics(self,
                           batcher,
                           save_plots=False,
                           image_name ='metrics.png',
                           eval_batch_size=50):

        """
        Note: This method only works for binary classification 
        as auPRC and auROC graphs only apply to binary classificaton problems.
    
        TODO: Modify this code to perform auROC generation
        for one-vs-all in the case of multiclass classification.

        """
                
        #Ref: http://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics
        ##auROC calculations

        #Keep batch size at 1 for now to ensure 1 full epoch is evaluated
    
    
        all_labels = np.zeros((batcher.num_records,self.num_classes), dtype = np.float32)
        all_probs = np.zeros((batcher.num_records,self.num_classes), dtype = np.float32)
       
        #num_correct = 0 #counts number of correct predictions
        num_whole_pulls =  batcher.num_records//eval_batch_size
        num_single_pulls = batcher.num_records%eval_batch_size
        num_steps = num_whole_pulls+num_single_pulls
        for i in range(num_steps):
            if i<num_whole_pulls:
                batch_size=eval_batch_size
            else:
                batch_size=1

            labels_batch, dna_seq_batch = batcher.pull_batch(batch_size)
            feed_dict = {
                          self.dna_seq_placeholder:dna_seq_batch,
                          self.labels_placeholder:labels_batch,
                          self.keep_prob_placeholder:1.0
                        }

                
            cur_prob= self.sess.run(self.probs,feed_dict=feed_dict)
            
            
                        
            #Fill labels array
            if batch_size > 1:
                start_ind = batch_size*i
            elif batch_size == 1:
                start_ind = num_whole_pulls*eval_batch_size+(i-num_whole_pulls)
            else:
                print "Never reach this condition"


            all_labels[start_ind:start_ind+batch_size,:] =  labels_batch
            all_probs[start_ind:start_ind+batch_size,:]  = cur_prob

        
        #Calculate metrics and save results in a dict
        md = self.calc_classifier_metrics(all_labels,all_probs)
        md["epoch"]=self.epoch
        md["step"]=self.step        

        #print "Testing accuracy",float(num_correct)/float(batcher.num_records)
        
        print 'Num examples: %d  Num correct: %d  Accuracy: %0.04f' % \
                  (batcher.num_records, md["num_correct"], md["accuracy"])+'\n'

      

        if save_plots:
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

            ax1[0].plot(md["fpr"],md["tpr"],color=plot_colors.next(),
                        lw=2,linestyle='-',label='auROC curve (area=%0.2f)' % md["auroc"] )


            #Generate auPRC plot axes
            #ax1[1].plot([0,1],[1,1],color='royalblue',lw=2,linestyle='--')
            ax1[1].set_xlabel('Precision')
            ax1[1].set_ylabel('Recall')
            ax1[1].set_title('auPRC')
            ax1[1].plot(md["thresh_precision"],md["thresh_recall"],color=plot_colors.next(),
                        lw=2,linestyle='-',label='auPRC curve (area=%0.2f)' % md["auprc"] )
    
            ax1[1].set_xbound(0.0,1.0)
            ax1[1].set_ybound(0.0,1.05)
           
            #Note: avg prec score is the area under the prec recall curve

            #Note: Presumably class 1 (pos examples) should be the only f1 score we focus on
            #print "F1 score for class",i,"is",f1_score
            plt.tight_layout()
        
            plt_fname = self.save_dir+os.sep+image_name
            print "Saving auROC image to",plt_fname
            fig1.savefig(plt_fname)

        #Return metrics dictionary
        return md


    
    def calc_classifier_metrics(self,all_labels,all_probs):
        """Calculate some metrics for the dataset
           return dictionary with metrics

        :param all_probs: nx2 prob values
        :param all_labels: nx2 labels
        :returns: dictionary of metrics
        :rtype: dict()

        """
        num_records = all_probs.shape[0]
        all_preds = np.zeros((num_records, self.num_classes),dtype = np.float32)
        all_preds[np.arange(num_records),all_probs.argmax(1)] = 1


        #Calculate accuracy
        num_correct = metrics.accuracy_score(all_labels[:,self.pos_index],all_preds[:,self.pos_index],normalize=False)
        accuracy = num_correct/float(all_preds.shape[0])
       
        
        ###Calculate auROC
        #http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        #metrics.roc_curve(y_true, y_score[, ...]) #y_score is probs

        fpr,tpr,_ = metrics.roc_curve(all_labels[:,self.pos_index],
                                      all_probs[:,self.pos_index],
                                      pos_label=self.pos_index)
        auroc = metrics.auc(fpr,tpr)
        
        thresh_precision,thresh_recall,prc_thresholds = metrics.precision_recall_curve(
                                                        all_labels[:,self.pos_index],
                                                        all_probs[:,self.pos_index])
        
        #Calculate precision, recall, and f1-score for threshold = 0.5
        #confusion_matrix = metrics.confusion_matrix(all_labels[:,self.pos_index],all_probs[:,self.pos_index])
        precision, recall, f1_score, support = metrics.precision_recall_fscore_support(
                                                all_labels[:,self.pos_index],
                                                all_preds[:,self.pos_index],
                                                pos_label=self.pos_index)

        precision = precision[self.pos_index]
        recall = recall[self.pos_index]
        f1_score = f1_score[self.pos_index]
        support = support[self.pos_index]
        
        auprc = metrics.average_precision_score(all_labels[:,self.pos_index],
                                                all_probs[:,self.pos_index])
        
        return OrderedDict([
                ("num_correct",num_correct),
                ("accuracy",accuracy), 
                ("auroc",auroc),
                ("auprc",auprc),
                ("fpr",fpr),
                ("tpr",tpr),
                ("precision",precision),
                ("recall",recall),
                ("f1_score",f1_score),
                ("support",support),
                ("thresh_precision",thresh_precision),
                ("thresh_recall",thresh_recall),
                ("prc_thresholds",prc_thresholds)
                 ])


    
      
    
