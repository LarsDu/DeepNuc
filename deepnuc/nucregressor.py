import tensorflow as tf
import sklearn.metrics as metrics
import numpy as np

import nucconvmodel

import sys
import os
sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from duseqlogo import LogoTools

import scipy.stats

from nucinference import NucInference

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pprint
from itertools import cycle

import nucheatmap
from collections import OrderedDict


class NucRegressor(NucInference):

    use_onehot_labels = False
    
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
                 classification_threshold=None,
                 output_scale=[0,1]):

        super(NucRegressor, self).__init__( sess,
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
                                    nn_method_key)


        self.training_mean,self.training_std = self.train_batcher.get_label_mean_std()
        self.training_min,self.training_max = self.train_batcher.get_label_min_max()
        '''
        print "Training mean:\t",self.training_mean
        print "Training standard deviation:\t",self.training_std
        print "Training min:\t", self.training_min
        print "Training max:\t",self.training_max
        '''
        
        #For now this can only work if nucdata and batcher are specified as having 1 class
        if self.train_batcher.num_classes != 1:
            print "Error, more than two classes detected in train batcher"
        else:
            self.num_classes = 1
            
   
        self.classification_threshold = classification_threshold

        #In order to plot figures in their original scale instead of [0,1] we need to
        #pass the nuc regressor, the original scale the data came in.
        self.output_scale = output_scale
        self.save_on_epoch = 50


        

    def build_model(self):

        self.dna_seq_placeholder = tf.placeholder(tf.float32,
                                          shape=[None,self.seq_len,4],
                                          name="dna_seq")

        #Note for regression, these perhaps should not be called labels
        self.labels_placeholder = tf.placeholder(tf.float32,
                                               shape=[None, self.num_classes],
                                            name="labels")

        self.keep_prob_placeholder = tf.placeholder(tf.float32,name="keep_prob")

        #Note: Since I am not using a sigmoid here, technically these are not logits
        self.raw_logits, self.network = self.nn_method(self.dna_seq_placeholder,
                                                   self.keep_prob_placeholder,
                                                   self.num_classes)

        self.logits = tf.nn.sigmoid(self.raw_logits)
        #self.descaled_prediction = (self.logits * self.training_std)+self.training_mean
        #self.descaled_prediction = self.logits*(self.training_max-self.training_min)+self.training_min

        #self.standardized_labels = (self.labels_placeholder-self.training_mean)/self.training_std
        #self.standardized_labels = (self.labels_placeholder-self.training_min)/(self.training_max-self.training_min)
        
        #Regression tasks should use mean squared error
        self.squared_diff = tf.squared_difference(self.logits,self.labels_placeholder)
        self.loss = tf.reduce_mean(self.squared_diff)
    
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
        self.sess.run(self.init_op)
        self.load(self.checkpoint_dir)
        
        
    def eval_model_metrics(self,
                           batcher,
                           save_plots=False,
                           image_name='auroc_auprc.png',
                           eval_batch_size=50):
        """
        Note: This method is intended to only be used for regression tasks 
        """
        
        all_true = np.zeros((batcher.num_records,self.num_classes), dtype = np.float32)
        all_preds = np.zeros((batcher.num_records,self.num_classes), dtype = np.float32)
        
        
        num_whole_pulls =  batcher.num_records//eval_batch_size
        num_single_pulls= batcher.num_records%eval_batch_size
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

                
            cur_preds= self.sess.run(self.logits,feed_dict=feed_dict)

           
            #Fill labels array
            if batch_size > 1:
                start_ind = batch_size*i
            elif batch_size == 1:
                start_ind = num_whole_pulls*eval_batch_size+(i-num_whole_pulls)
            else:
                print "Never reach this condition"

            all_true[start_ind:start_ind+batch_size] =  labels_batch
            all_preds[start_ind:start_ind+batch_size]  = cur_preds
            
            
        '''
        #Test code
        print "Classification threshold",self.classification_threshold
        print "Checking here", np.sum(all_true>self.classification_threshold).astype(np.float32)
        print "DOUBLE checking"
        print all_true.shape
        counter = 0
        for i in range(batcher.num_records):
            lab,_ = batcher.pull_batch(1)
            if lab>self.classification_threshold:
                counter += 1
        print "Counter",counter
        '''

        print "True",all_true[-1:-10,0]
        print "Preds",all_preds[-1:-10,0]
            
        #Calc metrics and save results in a dict
        md = self.calc_regression_metrics(all_true[:,0],all_preds[:,0])
        md["epoch"]=self.epoch
        md["step"]=self.step        
        print 'Mean Absolute Error: %0.08f  Mean Squared Error: %0.08f Median Absolute Error: %0.08f  R-squared score %0.08f' % (md["mean_absolute_error"],
                             md["mean_squared_error"],
                             md["median_absolute_error"],
                             md["r2_score"])
               
        

        
        print 'Pearson correlation: %0.04f  Pearson p-value: %0.04f \
               Spearman correlation: %.04f  Spearman p-value: %.04f' % \
                                           (md['pearson_correlation'],
                                           md['pearson_pvalue'],
                                           md['spearman_correlation'],
                                           md['spearman_pvalue'])+"\n"


        #print "Classification threshold!",self.classification_threshold
        if self.classification_threshold:
            '''
            If a classification threshold was specified, calc
            auROC,auPRC and other classification metrics
            '''
            cd = self.calc_thresh_classifier_metrics(all_true,
                                                    all_preds,
                                                    self.classification_threshold)

            #Add cd dict entries to md
            md.update(cd)

            num_correct = md["accuracy"]
            print 'Num examples: %d  Num correct: %d  Accuracy: %0.04f' % \
                  (batcher.num_records, md["num_correct"], md["accuracy"])+'\n'
      
        
                
        if save_plots and self.classification_threshold:
             ###Plot some metrics
            plot_colors = cycle(['cyan','blue','orange','teal'])
        
    
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
        
        return md


    def calc_thresh_classifier_metrics(self,all_true,all_preds,threshold):
        """
        Mark all predictions exceeding threshold as positive, and all
        
        """

        self.pos_index=1
        print "Thresh is",threshold
        binary_true = (all_true>threshold).astype(np.float32)
        binary_preds = (all_preds>threshold).astype(np.float32)
        fpr,tpr,_ = metrics.roc_curve( binary_true,
                                       binary_preds)
        auroc = metrics.auc(fpr,tpr)

        '''
        print "Raw inputs(all_true)",all_true
        print "Raw logits(all_probs)",all_preds
        print "Threshold labels on inputs",binary_true
        print "Sum of all thresholded inputs",np.sum(binary_true)
        print "Thresholded labels on logits",binary_preds
        print "Sum of thresholded logits",np.sum(binary_preds)
        '''
        thresh_precision,thresh_recall,prc_thresholds = metrics.precision_recall_curve(
                                                                            binary_true,
                                                                            all_preds,
                                                                            pos_label=self.pos_index)
        
        precision, recall, f1_score, support = metrics.precision_recall_fscore_support(
                                                binary_true,
                                                binary_preds,
                                                 pos_label=self.pos_index)
                                                

        num_correct = metrics.accuracy_score(binary_true,binary_preds,normalize=False)
        accuracy = num_correct/float(all_preds.shape[0])
        
        precision = precision[self.pos_index]
        recall = recall[self.pos_index]
        f1_score = f1_score[self.pos_index]
        support = support[self.pos_index]
        
        auprc = metrics.average_precision_score(binary_true,all_preds)
        
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


    def calc_regression_metrics(self,all_true,all_preds):
        mean_absolute_error = metrics.mean_absolute_error(all_true,all_preds)
        mean_squared_error = metrics.mean_squared_error(all_true,all_preds)
        median_absolute_error = metrics.median_absolute_error(all_true,all_preds)
        r2_score = metrics.r2_score(all_preds,all_preds)
        pearson_corr,pearson_pval = scipy.stats.pearsonr(all_true,all_preds)
        spear_corr,spear_pval = scipy.stats.spearmanr(all_true,all_preds)
    
        return OrderedDict([
            ("mean_absolute_error",mean_absolute_error),
            ("mean_squared_error",mean_squared_error),
            ("median_absolute_error",median_absolute_error),
            ("r2_score",r2_score),
            ("pearson_correlation",pearson_corr),
            ("pearson_pvalue",pearson_pval),
            ("spearman_correlation",spear_corr),
            ("spearman_pvalue",spear_pval)
                           ])



    
