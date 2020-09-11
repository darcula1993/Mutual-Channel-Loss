import random
import tensorflow as tf
import numpy as np

def MutualChannelLoss(y_true,x,channel_per_class = 3,beta = 10.0):
    batch_shape = 128
    num_class = 3
    
    def Mask(channel_per_class = 240):
        foo = [1.0] * int(np.ceil(channel_per_class // 2)) + [0.0] * int(np.floor(channel_per_class // 2))
        bar = []
        for i in range(num_class):
            random.shuffle(foo)
            bar += foo
        bar = [bar for i in range(batch_shape)]
        bar = np.array(bar).astype("float32")
        bar = bar.reshape(batch_shape,1,1,num_class * channel_per_class)
        bar = tf.constant(bar)
        return bar

    branch = x    
    branch = tf.reshape(branch,[-1,branch.shape[1] * branch.shape[2],branch.shape[3]]) # shape[0]:batch shape;
    branch = tf.nn.softmax(branch,axis = 1)
    branch = tf.reshape(branch,[-1,x.shape[1], x.shape[2],x.shape[3]])
    branch = tf.transpose(branch,perm = [0,3,2,1]) # (batch,channel,width,height)
    branch = tf.nn.max_pool2d(branch,ksize=(channel_per_class,1), strides=(channel_per_class,1),padding = "VALID")
    branch = tf.transpose(branch,perm = [0,3,2,1]) # (batch,height,width,channel)   
    branch = tf.reshape(branch,[-1,branch.shape[1] * branch.shape[2],branch.shape[3]])
    loss_2 = 1.0 - 1.0 * tf.reduce_mean(tf.reduce_sum(branch,axis = 1)) / channel_per_class

    mask = Mask(channel_per_class)
    branch_1 = x * mask
    branch_1 = tf.transpose(branch_1,perm = [0,3,2,1]) # (batch,channel,width,height)
    branch_1 = tf.nn.max_pool2d(branch_1,ksize=(channel_per_class,1), strides=(channel_per_class,1),padding = "VALID")
    branch_1 = tf.transpose(branch_1,perm = [0,3,2,1]) # (batch,height,width,channel)
    branch_1 = tf.keras.layers.GlobalAveragePooling2D()(branch_1)
    loss_1 = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_true,branch_1)
    loss_1 = tf.nn.compute_average_loss(loss_1, global_batch_size=constants.GENERATOR_BATCH_SIZE)
    mc_loss = loss_1 + beta * loss_2

    return mc_loss
