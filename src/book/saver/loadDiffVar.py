import tensorflow as tf

v = tf.Variable(tf.zeros([1, 2], name="v"))
# 通过变量重命名直接读取变量
saver = tf.train.Saver({"v1": v})

with tf.Session() as sess:
    saver.restore(sess, "./Model/test.ckpt")
    print(sess.run(v))
