import tensorflow as tf

class MeanSquareLoss(object):
    def __init__(self, minibatch=True):
        self._minibatch = minibatch 

    def compute_loss(self, y_true, y_pred):
        normalize_y_pred = tf.nn.l2_normalize(y_true,0)        
        normalize_y_true = tf.nn.l2_normalize(y_pred,0)
        return tf.reduce_sum( 1- tf.multiply( normalize_y_pred, normalize_y_true), 1, keep_dims=True )
