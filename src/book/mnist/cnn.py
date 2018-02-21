import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

MODULE_PATH = "./Module/mnist-cnn.ckpt"
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

# MNIST 图片数据 张量
# mnist.train.image 共有60000行数据, 每个向量 28 *28 像素 784
x = tf.placeholder("float", [None, 28 * 28], name="x")


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W, name=None):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME",
                        name=name)


def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME",
                          name=name)


x_image = tf.reshape(x, [-1, 28, 28, 1], name="x_image")

# 第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32], name="W_conv1")
b_conv1 = bias_variable([32], name="b_conv1")

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name="h_conv1")
h_pool = max_pool_2x2(h_conv1, name="h_pool")

# 第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64], name="W_conv2")
b_conv2 = bias_variable([64], name="b_conv2")

h_conv2 = tf.nn.relu(conv2d(h_pool, W_conv2) + b_conv2, name="h_conv2")
h_pool2 = max_pool_2x2(h_conv2, name="h_poo2")

# 密集连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024], name="W_fc1")
b_fc1 = bias_variable([1024], name="b_fc1")

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64], name="h_pool2_flat")
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name="h_fc1")

# Dropout
keep_prob = tf.placeholder("float", name="keep_prob")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name="h_fc1_drop")

W_fc2 = weight_variable([1024, 10], name="W_fc2")
b_fc2 = bias_variable([10], name="b_fc2")

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name="y_conv")

# 定义损失函数, 交叉熵 cross-entropy
y_ = tf.placeholder("float", [None, 10], name="y_")
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv), name="cross_entropy")

train_step = tf.train.AdamOptimizer(
    1e-4).minimize(cross_entropy, name="train_step")

correct_prediction = tf.equal(
    tf.argmax(y_conv, 1), tf.argmax(y_, 1), name="correct_prediction")
accuracy = tf.reduce_mean(
    tf.cast(correct_prediction, "float"), name="accuracy")

# 初始化变量
init = tf.global_variables_initializer()

# 声明 sess 并初始化
sess = tf.Session()
sess.run(init)

for variables in tf.global_variables():
    print("variables:", variables.name)

saver = tf.train.Saver()

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    sess.run(train_step, feed_dict={
             x: batch[0], y_: batch[1], keep_prob: 0.5})

# 验证集也进行切分
for i in range(20):
    test_set = mnist.test.next_batch(50)
    print("test accuracy %g" % sess.run(accuracy, feed_dict={
        x: test_set[0], y_: test_set[1], keep_prob: 1.0}))

saver.save(sess, MODULE_PATH)
sess.close()
