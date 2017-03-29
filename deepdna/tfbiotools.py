import tensorflow as tf

def prob_to_logit(p):
    return tf.log(p/(1-p))
