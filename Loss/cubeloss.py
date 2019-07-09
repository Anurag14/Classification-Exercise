import tensorflow as tf

class CubeLoss(object):
    def __init__(self, minibatch=True):
        self._minibatch = minibatch 

    def compute_loss(self, y_true, y_pred):
        # this code works for both batch and single points 
        square=tf.multiply(y_pred-y_true,y_pred-y_true)
        cubeloss=tf.reduce_sum(tf.multiply(y_pred-y_true,square))
        return cubeloss
        
