import tensorflow as tf


def get_weight(shape, lambda, name):
    # 生成一个变量
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32, name=name)
    # add_to_collection 函数将这个新生成的变量的L2正则化损失项加入集合
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(lambda)(var))
    return var


with tf.name_scope("input")
    x = tf.placeholder(tf.float32, [None, 2], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, 1], name="y-input")

batch_size = 8
