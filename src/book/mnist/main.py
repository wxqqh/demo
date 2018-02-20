import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

# 定义计算图

# 输入变量集
x = tf.placeholder(tf.float32, [None, 28 * 28], name="x_image")

# 权重
W = tf.Variable(tf.zeros([28*28, 10]), name="weight")

# 偏置量
b = tf.Variable(tf.zeros([10]), name="bias")
# fn = W*x + b
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 损失函数, 交叉熵
y_ = tf.placeholder(tf.float32, [None, 10], name="y_")
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session(tf.ConfigProto(allow_soft_placement=True,
                               log_device_placement=True)) as sess:
    sess.run(init)
    saver = tf.train.Saver()

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print(sess.run(accuracy, feed_dict={
          x: mnist.test.images, y_: mnist.test.labels}))

    saver.save(sess, "./Model/train")

    with tf.summary.FileWriter("./log", tf.get_default_graph()) as writer:
        print("log success")
