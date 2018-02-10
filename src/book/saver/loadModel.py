import tensorflow as tf

v1 = tf.Variable(tf.random_normal([1, 2]), name="v1")
v2 = tf.Variable(tf.random_normal([1, 2]), name="v2")

result = tf.add(v1, v2, name="add")

saver = tf.train.Saver()

with tf.Session() as sess:
    # 加载TensorFlow模型
    saver.restore(sess, "./Model/test.ckpt")
    print("result:", sess.run(result))
