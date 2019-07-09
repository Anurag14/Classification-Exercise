import tensorflow as tf

class MeanSquareLoss(object):
    def __init__(self, minibatch=True):
        self._minibatch = minibatch 

    def compute_loss(self, y_true, y_pred):
        # this tensorflow construct caters for batch size and single data both
        return tf.reduce_sum( tf.multiply( y_true - y_pred, y_true - y_pred), 1, keep_dims=True )
