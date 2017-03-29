import tensorflow as tf
import numpy as np
import DuBioTools as dbt
import DuNucInput
import NucInfModels
import relsaver
import matplotlib.pyplot as plt
import duseqlogo.LogoTools as LogoTools

#from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
import sklearn.metrics as metrics
from itertools import cycle

import os
import datetime
import time


import sys
from logger import Logger
from shutil import copyfile

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS


flags.DEFINE_string('json_params',"default.h5","""Full parameter file with files and training params""")
flags.DEFINE_string('mode',"train","""Valid modes are 'train','validate', 'eval', and 'decompose.' For eval and decompose mode, only checkpoints created from 'train' training runs are evaluated or decomposed.""")

flags.DEFINE_string('decomp_file','',"""A fasta or bed file with sequences to be evaluated or decomposed. Saves decompositions in an hdf5 file, and includes a reference file. Note that this can be run for either 'train' or 'validate' mode, and will decompose based on k=0 model  """)



def main(_):
    
    mode = FLAGS.mode
    params = DuNucInput.JsonParams(FLAGS.json_params)
    #num_iterations = params.num_iterations
    #learning_rate = params.learning_rate

    #Create save_dir in the same directory as the json_params file
    if not os.path.exists(params.save_dir):
        os.makedirs(params.save_dir)
   
    #Set up a log file and save it in the specified save directory
    base_fname = os.path.splitext(os.path.basename(params.json_filename))[0]
    log_fname = params.save_dir+os.sep+base_fname+'_'+mode+'.log'
    sys.stdout= Logger(log_fname)

    
    
    #Copy the json_params file and save it to the specified save_dir
    base_json = os.path.basename(FLAGS.json_params)
    copy_target = params.save_dir+os.sep+base_json
    #TODO: make it so that this copy does not overwrite 
    print "Copying",FLAGS.json_params,"to",copy_target
    copyfile(FLAGS.json_params,copy_target)
    
    ###File loading and k-folds training options
    
    if mode == 'train':
        #TRAIN FULL DATASET
        #In training mode, the testing_set is the same as the training set
        
        training_set = DuNucInput.InputCollection(params.training_file)
        training_set.open()
        run_ops('train',training_set,None,params,k=0)
        training_set.close()
        
    elif mode == 'validate':
        #K-FOLDS MODEL VALIDATION 
        print "Performing k-folds model validation..."
        print "k=",params.k_folds

        training_set = DuNucInput.InputCollection(params.training_file)
        
        print "Test fraction = ", params.k_validation_test_frac
        #Divide training_set into valid_train_set[k] and valid_test_set[k]
        valid_train_set = []
        valid_test_set = []
        valid_train_size = int((1-params.k_validation_test_frac)*
                                   training_set.num_records)
        valid_test_size = int(training_set.num_records-valid_train_size)
        print "K-folds training set size:",valid_train_size
        print "K-folds testing set size:",valid_test_size
            

        all_indices = np.random.permutation(training_set.perm_indices)
        for k in range(params.k_folds):
            print "\n\n\n"
            print "Starting on k-fold number", k
            #Need separate constructors for each training run!
            valid_train_set.append(DuNucInput.InputCollection(params.training_file))
            valid_test_set.append(DuNucInput.InputCollection(params.training_file))

            #change perm indices for training and test set
            valid_test_set[k].set_perm_indices(all_indices[(valid_test_size*k):
                            (valid_test_size*(k+1))])
            valid_train_set[k].set_perm_indices(np.setdiff1d(all_indices,
                                  valid_test_set[k].perm_indices))
            valid_train_set[k].open()
            valid_test_set[k].open()
            #TODO: (optional) save perm indices of each k_folds run 
            run_ops('validate',valid_train_set[k],valid_test_set[k],params,k=k)
            valid_train_set[k].close()
            valid_test_set[k].close()
            print "Time of completion for k=",k,":",datetime.datetime.utcnow()

    elif mode == 'eval':
        #Construct auROC curve and collect other metrics for model under 'train' mode
        #TODO: This mode only works with a valid holdout set specified
        training_set = DuNucInput.InputCollection(params.training_file)
        eval_set = DuNucInput.InputCollection(params.eval_file)
        eval_set.open()
        run_ops(mode,training_set,eval_set,params,k=0,checkpoint_folder='train')
        eval_set.close()
        
    elif mode == 'decompose':
        
        #Decompose the file specified by FLAGS.decomp_file
        
        dfext= os.path.splitext(FLAGS.decomp_file)[1]
        if dfext == '.h5' or dfext =='.hdf5':
            decomp_set = DuNucInput.InputCollection(FLAGS.decomp_file)
        elif dfext == '.fa' or dfext  == '.fasta':
            decomp_set = DuNucInput.FastaReader(FLAGS.decomp_file,params.seq_len)
            #TODO: Generate header/index matching class
        elif dfext == '.bed':
            decomp_set = DuNucInput.CoordReader(FLAGS.decomp_file,params.seq_len)
        decomp_set.open()
        run_ops(mode,decomp_set,None,params,k=0,checkpoint_folder='train')
        decomp_set.close()

    elif mode == 'decompose_test':
        """
        Visualize decomposition on randomly selected examples from the
        original training set
        """
        decomp_set = DuNucInput.InputCollection(params.training_file)
        #decomp_set = DuNucInput.FastaReader(FLAGS.decomp_file,params.seq_len)
        #decomp_set = DuNucInput.CoordReader(FLAGS.decomp_file,params.seq_len)
        decomp_set.open()
        run_ops(mode,decomp_set,None,params,k=0)
        decomp_set.close()
    

    #Close Logger
    sys.stdout.close()



                
def run_ops(mode,training_set,testing_set,json_params,k=0,checkpoint_folder=None):
    if checkpoint_folder==None and mode != None:
        #Set default checkpoint folder with same name as current mode
        checkpoint_folder=mode 

    '''
    Run all ops

    Args:
    	- mode: 'training', 'validate', 'eval',or 'decompose'
        - training_set: The training InputCollection obj
        - testing_set: (Optional) test set against which to evaluate model efficacy
        - json_params: JsonParams obj with training hyperparameters
        - k: Current k_fold index (used to name checkpoint and summary directories)
        - checkpoint_folder: Folder in params.save_dir from which to load model checkpoint.
                             Defaults to folder with same name as mode
        	
    '''
    
    
    save_dir = json_params.save_dir
    checkpoint_dir = save_dir+os.sep+checkpoint_folder+os.sep+"checkpoint_k"+str(k)
    summary_dir = save_dir+os.sep+checkpoint_folder+os.sep+"summaries_k"+str(k)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    

    model_graph =tf.Graph()
    with model_graph.as_default(): #Set graph context
        #with graph.device('/gpu:0'):


        print "Custom model params: ",json_params.custom_args_dict
       
        dtconv = NucInfModels.NucConvModel(json_params.seq_len,
                                     training_set.num_classes,
                                     **json_params.custom_args_dict)
    
        ##Initialize placeholders
        dna_seq_placeholder = tf.placeholder(tf.float32,
                                          shape=[None,4,dtconv.seq_len],
                                          name="DNA_SEQ_DATA")

        labels_placeholder = tf.placeholder(tf.float32,
                                            shape=[None, dtconv.num_classes],
                                            name="LABELS")

        #Keep prob placeholder
        keep_prob_placeholder = tf.placeholder(tf.float32,name="keep_prob")

        
        logits,network = dtconv.inferenceA(dna_seq_placeholder,
                                    keep_prob_placeholder)

        
        #logits,network = dtconv.inferenceB_600bp(dna_seq_placeholder,
        #                            keep_prob_placeholder)
        
        #logits,network = dtconv.inferenceE(dna_seq_placeholder,
        #                            keep_prob_placeholder)

        #logits,network = dtconv.inferenceC_600bp_lite(dna_seq_placeholder,
        #                            keep_prob_placeholder)
       
        
        relevance = network.relevance_backprop(logits*labels_placeholder)
        
        probs = dtconv.logits_to_probs(logits) #For final eval
        loss = dtconv.loss(logits,labels_placeholder)

        #Add gradient ops to graph with learning rate
        train_op = dtconv.training_adam(loss,json_params.learning_rate)

        #Count the number of correct predictions
        eval_num_correct = dtconv.evaluation(logits,labels_placeholder)

        #Consolidate summary ops for entire graph
        #summary_op = tf.merge_all_summaries()
        summary_op = tf.summary.merge_all()
        
        
        with tf.Session() as sess:
            #init_op = tf.initialize_all_variables()
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            saver = tf.train.Saver()

            if (mode == 'train' or mode == 'validate'):
                ###TRAIN OR K-FOLDS VALIDATION MODE###
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print "Checkpoint restored"
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print "No checkpoint found on",checkpoint_dir

                print "Running training mode..."
                #summary_writer = tf.train.SummaryWriter(summary_dir,sess.graph)
                summary_writer = tf.summary.FileWriter(summary_dir,sess.graph)
                
                for step in range (json_params.num_iterations):
            
                    start_time = time.time()
                    
                    (dna_seq_batch,
                    labels_batch) = training_set.pull_batch_train(json_params.batch_size)

                    feed_dict={
                           dna_seq_placeholder:dna_seq_batch,
                           labels_placeholder:labels_batch,
                           keep_prob_placeholder:0.5}

                    _,loss_value,_ = sess.run([train_op,loss,logits],feed_dict=feed_dict)
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    duration = time.time()-start_time


                    # Write the summaries and print an overview fairly often.
                    if step % 1000 == 0:
                        # Print status to stdout.
                        print('Step %d: loss = %.4f (%.3f sec)' % (step,
                                                     loss_value,
                                                      duration))
                        #print logits_value
                        # Update the events file.
                        summary_str = sess.run(summary_op, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()

                    # Save a checkpoint and (maybe) evaluate the model
                    #    against train and test
                    if (step + 1) % 5000 == 0 or (step + 1) == json_params.num_iterations:

                        ckpt_name = "ckpt_"+mode
                        print "Saving on ",checkpoint_dir
                        saver.save(sess,checkpoint_dir+os.sep+ckpt_name, global_step=step)

                        # Evaluate the entire training set.
                        print('Training data eval:')
                        eval_model( sess,
                                training_set,
                                eval_num_correct,
                                dna_seq_placeholder,
                                labels_placeholder,
                                keep_prob_placeholder,
                                save_file=checkpoint_dir+os.sep+'train_eval.log' )

                        if testing_set != None and testing_set.num_records > 0:
                            print('Testing data eval:')
                            #eval_model_metrics( sess,
                            #        testing_set,
                            #        eval_num_correct,
                            #        probs,
                            #        dna_seq_placeholder,
                            #        labels_placeholder,
                            #        keep_prob_placeholder,
                            #        save_file = checkpoint_dir+os.sep+'auroc'+str(k)+'.png')
                            eval_model( sess,
                                testing_set,
                                eval_num_correct,
                                dna_seq_placeholder,
                                labels_placeholder,
                                keep_prob_placeholder,
                                save_file=checkpoint_dir+os.sep+'test_eval.log')

            
                            
            elif mode == 'eval':
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print "Checkpoint restored"
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print "No checkpoint found on",checkpoint_dir

                '''Evaluate an entire dataset and generate an auROC curve'''

                print 'Data evaluation/auROC construction:'
                eval_model_metrics( sess,
                            testing_set,
                            eval_num_correct,
                            probs,
                            dna_seq_placeholder,
                            labels_placeholder,
                            keep_prob_placeholder,
                            save_file=checkpoint_dir+os.sep+'eval_metrics.log')

                    

            elif mode == 'decompose':

                
                
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print "Checkpoint restored from", checkpoint_dir
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print "No checkpoint found on",checkpoint_dir



                #decomp_set = training_set #To avoid name confusion
                decomp_set = training_set
                print "Decomposing sequences in file",FLAGS.decomp_file
                num_records = decomp_set.num_records
                print "Number of records in decomposition file:",
                #Test decomposition code for a few examples

                #TODO: Decomposition should be able to operate on fasta input,
                # coord input, or hdf5 input based on extension.
                #TODO: Clean this section up. Should be able to use ConvertToHdf5 here

                print "Type name for relevance save file (valid extensions are \
                                                             '.txt', '.h5','.hdf5')"

                rel_save_fname = input()
                rel_save_fname = checkpoint_dir+os.sep+rel_save_fname
                rel_fext = os.path.splitext(rel_save_fname)[1]
                
                decomp_set.num_classes  
                if rel_fext == '.hdf5' or rel_fext == '.h5':
                    rel_saver = relsaver.Hdf5Saver(rel_save_fname,
                                                         decomp_set.seq_len)
                                                         
                elif rel_fext == '.txt':
                    rel_saver = relsaver.TextSaver(rel_save_fname,                                                                                     decomp_set.seq_len)
                                                         
                else:
                    print "Invalid file extension"
        
                
                #TODO: Write relevances to file
                rel_saver.open()
                for step in range(num_records):
                    
                    dna_decomp_batch,labels_decomp_batch = decomp_set.pull_batch_eval(1)
                    feed_dict= {dna_seq_placeholder:dna_decomp_batch,
                                labels_placeholder:labels_decomp_batch,
                                keep_prob_placeholder:1.0}
                    r_input = sess.run(relevance,
                                feed_dict=feed_dict)
                    #Write relevances and label to file
                    rel_saver.write(r_input,labels_decomp_batch[0])
                rel_saver.close()
                    
                
            elif mode == 'decompose_test':
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print "Checkpoint restored"
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print "No checkpoint found on",checkpoint_dir

                #image_dir = save_dir+os.sep+"img"

                logosheets=[]
                input_seqs=[]

                
                for step in range(5):
                    (dna_seq_batch,
                    labels_batch) = training_set.pull_batch_eval(1)

                    feed_dict={dna_seq_placeholder:dna_seq_batch,
                           labels_placeholder:labels_batch,
                           keep_prob_placeholder:1.0}
                    
                    
                    #Note that this should only hold if batch_size=1
                    #flat_relevance = tf.reshape(relevance,[-1])
                    r_input = sess.run(relevance,
                                           feed_dict=feed_dict)
                    r_img = np.transpose(np.squeeze(r_input[:,:,12:-12,:],axis=(0,1)))

                    r_slice = r_img[:,275:300]
                    np.set_printoptions(linewidth=500,precision=4)
                    #print r_slice
                    plt.pcolor(r_slice,cmap=plt.cm.Reds)
                    #plt.show()

                    #print "A relevance"
                    #plt.plot(r_img[0,:])
                    #plt.show()
                    #print "Relevance by position"
                    #plt.plot(np.sum(r_img,axis=0))
                    #plt.show()

                    
                    logits_np = sess.run(logits,
                                 feed_dict=feed_dict)

                    guess = logits_np.tolist()
                    guess = guess[0].index(max(guess[0]))
                    
                    actual = labels_batch[0].tolist().index(1.)

                    #print logits_np
                    print sess.run(probs,feed_dict=feed_dict)
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

                    
                rel_sheet = LogoTools.LogoNucSheet(logosheets,input_seqs,input_type='heights')
                rel_sheet.write_to_png('test_refactor.png')
                    
def eval_model(sess,
               input_collection,
               eval_num_correct,
               dna_seq_placeholder,
               labels_placeholder,
               keep_prob_placeholder,
               save_file=None):

    
    eval_batch_size = 1
    #Keep batch size at 1 for now to ensure 1 full epoch is evaluated

    num_correct=0 #counts number of correct predictions
    steps_per_epoch = input_collection.num_records//eval_batch_size
    
    for _ in range(steps_per_epoch):
        dna_seq_batch,labels_batch = input_collection.pull_batch_eval(eval_batch_size)
        feed_dict = {
                     dna_seq_placeholder:dna_seq_batch,
                     labels_placeholder:labels_batch,
                     keep_prob_placeholder:1.0
                    }
        num_correct += sess.run(eval_num_correct,feed_dict=feed_dict)

    frac_correct = float(num_correct)/input_collection.num_records #aka precision

    print_out = 'Num examples: %d  Num correct: %d  Precision: %0.04f' % (input_collection.num_records, num_correct, frac_correct)+'\n'
    
    

    if save_file != None:
        with open(save_file,'a') as tlgf:
            tlgf.write(print_out)
    
    
    #summary_writer.add_summary(loss+'/'+summar_tag,step)
    print (print_out)
    


def eval_model_metrics(sess,
               input_collection,
               eval_num_correct,
               probs,
               dna_seq_placeholder,
               labels_placeholder,
               keep_prob_placeholder,
               save_file=None,
               show_plot=True):

    '''
    Note: This method only works for binary classification
    as auPRC and auROC graphs only apply to binary classificaton problems.

    Additional note: In the future, modify this code to perform auROC generation
    for one-vs-all in the case of multiclass classification.
    
    '''
    
    #Ref: http://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics
    ##auROC calculations
    eval_batch_size = 1
    #Keep batch size at 1 for now to ensure 1 full epoch is evaluated

    num_correct=0 #counts number of correct predictions
    steps_per_epoch = input_collection.num_records//eval_batch_size

    all_labels = np.asarray([0,0],dtype=np.float32)#true labels
    all_probs = np.asarray([0,0],dtype=np.float32) #probabilities (scores)
    all_preds = np.asarray([0,0],dtype=np.float32) #predictions (highest scorers)
    pos_class_ind = 1

    for _ in range(steps_per_epoch):
    #for _ in range(20):
        dna_seq_batch,labels_batch = input_collection.pull_batch_eval(eval_batch_size)
        feed_dict = {
                     dna_seq_placeholder:dna_seq_batch,
                     labels_placeholder:labels_batch,
                     keep_prob_placeholder:1.0
                    }
        n_correct,cur_prob = sess.run([eval_num_correct,probs],feed_dict=feed_dict)
        num_correct += n_correct

        cur_pred_ind = np.argmax(cur_prob[0]) #index of largest prob
        cur_pred = np.zeros((1,2))
        cur_pred[0,cur_pred_ind]=1.
        
        #Stack append entries
        all_labels = np.vstack((all_labels,labels_batch[0]))
        all_probs = np.vstack((all_probs,cur_prob[0]))
        all_preds = np.vstack((all_preds,cur_pred))

    #Remove row 1 which was initialized as zeros
    all_labels = np.delete(all_labels,(0),axis=0)
    all_probs = np.delete(all_probs,(0),axis=0)
    all_preds = np.delete(all_probs,(0),axis=0)
    
    frac_correct = float(num_correct)/input_collection.num_records
    #http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html


    #dict keys correspond to available classes
     
    
    
    fpr = dict()
    tpr = dict()
    precision = dict()
    recall = dict()


    plot_colors = cycle(['cyan','blue','orange','teal'])

    #Generate auROC plot axes
    print "Labels shape",all_labels.shape
    print "Probs shape",all_probs.shape
    print "Preds shape",all_preds.shape
    fig1,ax1  = plt.subplots(2)
    
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
  

      
    for i in range(2):#i is class key
        #metrics.roc_curve(y_true, y_score[, ...]) #y_score is probs
        fpr[i],tpr[i],roc_thresholds = metrics.roc_curve(all_labels[:,i],all_probs[:,i])
        auc = metrics.auc(fpr[i],tpr[i])


        precision[i],recall[i],prc_thresholds = metrics.precision_recall_curve(all_labels[:,i],
                                                                               all_probs[:,i])
        avg_prec_score = metrics.average_precision_score(all_labels[:,i],all_probs[:,i])
        #f1_score = metrics.f1_score(all_labels[:,i],all_preds[:,i]) 
        
        
        ax1[0].plot(fpr[i],tpr[i],color=plot_colors.next(),
                lw=2,linestyle='-',label='ROC curve (area=%0.2f)' % auc )

        ax1[1].plot(precision[i],recall[i],color=plot_colors.next(),
                    lw=2,linestyle='-',label='PRC curve (area=%0.2f)' % avg_prec_score )

        print "AUC for class", i,"is",auc
        print "Average precision score for class",i,"is",avg_prec_score
        
        #Note: avg prec score is the area under the prec recall curve

        #Note: Presumably class 1 (pos examples) should be the only f1 score we focus on
        #print "F1 score for class",i,"is",f1_score
        
    
    print_out = 'Num examples: %d  Num correct: %d  Precision: %0.04f' % (input_collection.num_records, num_correct, frac_correct)+'\n'
        
    if show_plot:
        plt.show()
    if save_file != None:
        fig1_fname = os.path.splitext(save_file)[0]+'_metrics.png'
        print "Saving auROC image to",save_file
        fig1.savefig(save_file,bbox_inches='tight')
       
        #Log precision in text file for possibly making graphs later
        
        with open(save_file,'a') as tlgf:
            tlgf.write(print_out)
            tlgf.write("AUC for class", i,"is",auc,'\n')
            tlgf.write("Average precision score for class",i,"is",avg_prec_score,'\n')
       
    
    #summary_writer.add_summary(loss+'/'+summar_tag,step)
    print (print_out)
    

    

                
if __name__ == '__main__':
    tf.app.run()
