import tensorflow as tf
import numpy as np
import DuBioTools as dbt
import DuNucInput
import os

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS


flags.DEFINE_string('json_params',"","""Full parameter file with files and training params""")
flags.DEFINE_string('mode',"train","""Valid modes are 'train','validate', and 'eval'""")


def main(_):
    mode = FLAGS.mode
    params = DuNucInput.JsonParams(FLAGS.json_params)
    save_dir = params.save_dir

    #num_iterations = params.num_iterations
    #learning_rate = params.learning_rate
    
    
    
    checkpoint_dir = save_dir+"/checkpoints"
    summary_dir = save_dir+"/summaries"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    #TODO: Add file loading, k_folds options, and inference model here    

    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        saver = tf.train.Saver()

        if mode == 'train' or mode == 'validate':
            ###TRAIN OR K-FOLDS VALIDATION MODE###
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print "Checkpoint restored"
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print "No checkpoint found on",checkpoint_dir

            print "Running training mode..."
            summary_writer = tf.train.SummaryWriter(summary_dir,sess.graph)

            for i in range (params.num_iterations):
                if i%5000 == 0:
                    print i,"training iterations passed"
                #TODO: Insert sess.run() here
            #TODO: Insert accuracy code here
        elif mode == 'eval':
            ###K-FOLDS VALIDATION MODE
            pass
        




def eval_model(sess,
               input_collection,
               eval_num_correct,
               dna_seq_placeholder,
               keep_prob_placeholder):

    
    eval_batch_size = 1
    #Keep batch size at 1 for now to ensure 1 full epoch is evaluated
    
    steps_per_epoch = input_collection.num_records//eval_batch_size
    
    for _ in range(steps_per_epoch):
        dna_seq_batch,labels_batch = input_collection.pull_batch_eval(eval_batch_size)

        feed_dict = {
                     dna_seq_placeholder:dna_seq_batch,
                     labels_placeholder:labels_batch
                    }


            
        num_correct += sess.run(eval_num_correct,feed_dict=feed_dict)

    precision = float(num_correct)/data_collection.num_examples

    #summary_writer.add_summary(loss+'/'+summar_tag,step)
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (data_collection.num_examples, num_correct, precision))


                
if __name__ == '__main__':
    tf.app.run()
