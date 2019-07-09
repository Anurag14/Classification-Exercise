import tensorflow as tf

class NormalizeCubeLoss(object):
    def __init__(self, minibatch=True):
        self._minibatch = minibatch 

    def compute_loss(self, y_true, y_pred):
        if not self._minibatch:
            normalize_y_pred = tf.nn.l2_normalize(y_true,0)        
            normalize_y_true = tf.nn.l2_normalize(y_pred,0)
            square=tf.multiply(normalize_y_pred-normalize_y_true,normalize_y_pred-normalize_y_true)
            normcubeloss=tf.reduce_sum(tf.multiply(normalize_y_pred-normalize_y_true,square))
        else:
            normalize_y_pred = tf.nn.l2_normalize(y_pred,1)        
            normalize_y_true = tf.nn.l2_normalize(y_true,1)
            square=tf.multiply(normalize_y_pred-normalize_y_true,normalize_y_pred-normalize_y_true)
            normcubeloss=tf.reduce_sum(tf.multiply(normalize_y_pred-normalize_y_true,square))
        return normcubeloss
