import tensorflow as tf
import sklearn.metrics as metrics
import numpy as np

import nucconvmodel

import sys
import os.path
sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from duseqlogo import LogoTools

import scipy.stats

from nucinference import NucInference

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
                 concat_revcom_input=False):

        
        self.sess = sess
        self.train_batcher = train_batcher
        self.test_batcher = test_batcher

        self.training_mean,self.training_std = self.train_batcher.get_label_mean_std()
        self.training_min,self.training_max = self.train_batcher.get_label_min_max()
        print "Training mean:\t",self.training_mean
        print "Training standard deviation:\t",self.training_std
        print "Training min:\t", self.training_min
        print "Training max:\t",self.training_max

        
        #For now this can only work if nucdata and batcher are specified as having 1 class
        if self.train_batcher.num_classes != 1 or self.test_batcher.num_classes !=1:
            print "Error, more than two classes detected in batchers"
        else:
            self.num_classes = 1
            
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

        #This flag will tell the inference method to concatenate
        #the reverse complemented version of the input sequence
        #to the input vector
        self.concat_revcom_input = concat_revcom_input
                
        self.train_steps_per_epoch = int(self.train_batcher.num_records//self.batch_size)
        self.test_steps_per_epoch = int(self.test_batcher.num_records//self.batch_size)
        self.total_iterations = int(self.train_steps_per_epoch*self.num_epochs)




    def build_model(self,nn_method):

        self.nn_method = nn_method
        self.dna_seq_placeholder = tf.placeholder(tf.float32,
                                          shape=[None,self.seq_len,4],
                                          name="dna_seq")

        #Note for regression, these perhaps should not be called labels
        self.labels_placeholder = tf.placeholder(tf.float32,
                                               shape=[None, self.num_classes],
                                            name="labels")

        self.keep_prob_placeholder = tf.placeholder(tf.float32,name="keep_prob")

        self.logits, self.network = self.nn_method(self.dna_seq_placeholder,
                                                   self.keep_prob_placeholder,
                                                   self.num_classes)


        #self.descaled_prediction = (self.logits * self.training_std)+self.training_mean
        self.descaled_prediction = self.logits*(self.training_max-self.training_min)+self.training_min

        self.probs = tf.nn.softmax(self.logits)


        #self.standardized_labels = (self.labels_placeholder-self.training_mean)/self.training_std
        self.standardized_labels = (self.labels_placeholder-self.training_min)/(self.training_max-self.training_min)
        
        #Regression tasks should use mean squared error
        self.loss = tf.reduce_mean(tf.squared_difference(self.logits,self.standardized_labels))

        # Add gradient ops to graph with learning rate
        self.train_op = tf.train.AdamOptimizer(self.learning_rate,beta1=self.beta1).\
                                                         minimize(self.loss)
    
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



    
    def eval_model_metrics(self,batcher,show_plots=False,save_plots=False,eval_batch_size=50):
        """
        Note: This method is intended to only be used for regression tasks 
        """
        
        all_true = np.zeros((batcher.num_records,self.num_classes), dtype = np.float32)
        all_preds = np.zeros((batcher.num_records,self.num_classes), dtype = np.float32)
       
        
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

                
            cur_pred= self.sess.run([self.descaled_prediction],feed_dict=feed_dict)
            
            #Fill labels array
            all_true[i:i+batch_size,:] =  labels_batch[0]
            all_preds[i:i+batch_size,:]  = cur_pred[0]

            

        #Calculate metrics and save results in a dict
        md = self.calculate_regression_metrics(all_true[:,0],all_preds[:,0])

                
        print 'Mean Absolute Error: %0.04f  Mean Squared Error: %0.04f \
               Median Absolute Error: %0.04f  R-squared score %0.04f' % (md["mean_absolute_error"],
                                                                         md["mean_squared_error"],
                                                                       md["median_absolute_error"],
                                                                       md["r2_score"])
               
        print 'Pearson correlation: %0.04f  Spearman correlation: %0.04f \
               Spearman correlation: %.04f  Spearman P-value: %.04f' % \
                                           (md['pearson_correlation'],
                                           md['pearson_pvalue'],
                                           md['spearman_correlation'],
                                           md['spearman_pvalue'])+"\n"
        


        
        if show_plots:
            pass
        if save_plots:
            pass

        return md


    def calculate_regression_metrics(self,all_true,all_preds):
        mean_absolute_error = metrics.mean_absolute_error(all_true,all_preds)
        mean_squared_error = metrics.mean_squared_error(all_true,all_preds)
        median_absolute_error = metrics.median_absolute_error(all_true,all_preds)
        r2_score = metrics.r2_score(all_preds,all_preds)
        pearson_corr,pearson_pval = scipy.stats.pearsonr(all_true,all_preds)
        spear_corr,spear_pval = scipy.stats.spearmanr(all_true,all_preds)
    
        return {"mean_absolute_error":mean_absolute_error,
                "mean_squared_error":mean_squared_error,
                "median_absolute_error":median_absolute_error,
                "r2_score":r2_score,
                "pearson_correlation":pearson_corr,
                "pearson_pvalue":pearson_pval,
                "spearman_correlation":spear_corr,
                "spearman_pvalue":spear_pval}



    
