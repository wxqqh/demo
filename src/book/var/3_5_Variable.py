import tensorflow as tf
# 实现一个神经网络的向前传播

# 声明传入特征向量 1*2 的矩阵
a = tf.constant([[.7, .9]], name="a")
# 声明一个[2*3] 的权重矩阵
w1 = tf.Variable(tf.random_normal([2, 3], mean=1, stddev=1, seed=1), name="w1")

# 声明一个[1*3] 的权重矩阵
w2 = tf.Variable(tf.random_normal([3, 1], mean=1, stddev=1, seed=1), name="w2")

x1 = tf.matmul(a, w1, name="x1")

y = tf.matmul(x1, w2, name="y")

saver = tf.train.Saver()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, "./model/train")
    writer = tf.summary.FileWriter("./log", graph=tf.get_default_graph())
    writer.close()
    print(sess.run(y))
