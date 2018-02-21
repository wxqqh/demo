import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# MNIST 图片数据 张量
# mnist.train.image 共有60000行数据, 每个向量 28 *28 像素 784
x = tf.placeholder(tf.float32, [None, 28 * 28], name="x_image")

# 权重值, 为了得到某个图片数据属于某个特定数字类的证据, 需要对图片像素值进行加权求和
# 如果这个像素具有很强的证据说明这个图片属于该类, 相关的权值为正数
W = tf.Variable(tf.zeros([28*28, 10]), name="weight")

# 偏置量, 修正值 防止过拟合
b = tf.Variable(tf.zeros([10]), name="bias")

# 判断图片->数字的概率
y = tf.nn.softmax(tf.matmul(x, W) + b)  # 0.9085
# 这里不应该是操作符重载么? 为啥会对结果有影响?
# y = tf.nn.softmax(tf.add(tf.matmul(x, W), b)) # 0.9136
# 黑人问号???
# y = tf.nn.softmax(tf.matmul(x, W))  # 0.9172

# 定义损失函数, 交叉熵 cross-entropy
y_ = tf.placeholder(tf.float32, [None, 10], name="y_")
cross_entropy = - tf.reduce_sum(y_ * tf.log(y))

# tf会自动地使用反向传播算法(backpropagation algorithm)来有效地确定变量是如何影响要最小化的那个成本值的
# 用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵。
# 梯度下降算法（gradient descent algorithm）是一个简单的学习过程，
# TensorFlow只需将每个变量一点点地往使成本不断降低的方向移动
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化变量
init = tf.global_variables_initializer()

# 声明 sess 并初始化
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()
# 训练1000次
for i in range(1000):
    # 使用一小部分的随机数据来进行训练被称为随机训练(stochastic training)
    # 在这里更确切的说是随机梯度下降训练。
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 评估模型

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 验证集也进行切分
for i in range(20):
    test_set = mnist.test.next_batch(50)
    print("test accuracy %g" % sess.run(accuracy, feed_dict={
      x: mnist.test.images, y_: mnist.test.labels}))

saver.save(sess, "./Model/train")
with tf.summary.FileWriter("./log", tf.get_default_graph()) as writer:
    print("log success")

sess.close()