import tensorflow as tf

# 不重复定义计算图上的运算，直接加载已经持久化的图

saver = tf.train.import_meta_graph("./Model/test.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess, "./Model/test.ckpt")
    print("v1:", sess.run(tf.get_default_graph().get_tensor_by_name("v1:0")))
