import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

def clip_by_local_norm(gradients, norm):
    for idx, grad in enumerate(gradients):
        gradients[idx] = tf.clip_by_norm(grad, norm)
    return gradients