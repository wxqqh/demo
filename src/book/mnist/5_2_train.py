import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST 数据集相关的常量
INPUT_NODE = 784  # 输入层节点数, 图片像素 28 * 28
OUPUT_NODE = 10  # 输出层节点数, 0~9 一共 10 个数字

# 配置神经网络参数
LAYER1_NODE = 500  # 隐藏层节点数, 目前只使用一层隐藏层作为尝试
BATCH_SIZE = 100  # 一个训练batch中的训练数据个数. 数字越小, 越接近随机梯度下降. 数字越大, 越接近梯度下降

LEARNING_RATE_BASE = 0.8  # 基础的学习率
LEARNING_RATE_DEACY = 0.99  # 学习率的衰减率

REGULARIZATION_RAGE = 0.0001  # 描述模型负责度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000  # 训练轮数

MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率

# 辅助函数, 给定神经网络的输入和所有参数, 计算神经网络的向前传播结果
# 在这里定义了一个使用ReLU激活函数的三层全连接神经网络, 通过加入隐藏层实现了多层网络结构
# 通过ReLU激活函数实现了去线性化, 在这个函数也通过支持传入用于计算参数平均值的类
# 这样方便在测试时候使用滑动平均模型


def inference(input_tensor, avg_class, weight1, biase1, weight2, biase2):
    if avg_class is None:
        # 计算隐藏层的向前传播结果 f(x) = x * M + biase
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weight1) + biase1)
        # 计算输出层的向前传播结果, 因为在计算损失函数时会一并计算softmax函数
        # 所以这里不需要加入激活函数, 而且不加入softmax不会影响预测结果
        # 因为在与测试使用的是不同类别对应节点输出值的相对大小, 有没有softmax层对最后分类的结果没有影响
        # 玉石在计算整个神经网络向前传播的时候可以不加入最后的softmax层
        return tf.matmul(layer1, weight2) + biase2
    else:
        # 首先使用avg_class.agerage函数来计算得出变量的滑动平均值
        # 然后在计算相应的神经网络向前传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.agerage(
            weight1)) + avg_class.agerage(biase1))
        return tf.nn.relu(tf.matmul(layer1, avg_class.agerage(
            weight2)) + avg_class.agerage(biase2))


# 训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, OUPUT_NODE], name="y-input")

    # 生成隐藏层的参数
    weight1 = tf.Variable(tf.truncated_normal(
        [INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biase1 = tf.Variable(tf.constant(0.1, [LAYER1_NODE]))
    # 生成输出层的参数
    weight2 = tf.Variable(tf.truncated_normal(
        [LAYER1_NODE, OUPUT_NODE], stddev=0.1))
    biase2 = tf.Variable(tf.constant(0.1, [OUPUT_NODE]))

    # 计算在当前参数下神经网络向前传播的结果.
    y = inference(x, None, weight1, biase1, weight2, biase2)

    # 定义存储训练轮数的变量. 这个变量不需要计算滑动平均值
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量, 初始化滑动平均类
    # 给定训练轮数的变量可以加快训练早起变量的更新速度

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均,
    # tf.trainable_variables返回的就是图上集合, GraphKeys.TRAINABLE_VARIABLES中的元素
    # 这个集合的元素就是所有没有指定trainable=False的参数。

    variable_averages_op = variable_averages.apply(tf.trainable_variables)

    # 当需要使用这个滑动平均值的时, 需要明确调用average函数
    average_y = inference(x, variable_averages, weight1,
                          biase1, weight2, biase2)

    # 计算交叉熵作为刻画预测值和真实值之间的差距的损失函数.
    # 供的sparse_softmax_cross_entropy_with_logits函数来计算交叉熵
    # 当分类问题只有一个正确答案时, 可以使用这个函数来加速交叉熵的计算
    # MNIST问题的图片中只包含了0~9中的一个数字，所以可以使用这个函数来计算交叉熵损失
    # 这个函数的第一个参数是神经网络不包括softmax层的前向传播结果，第二个是训练数据的正确答案
    # 因为标准答案是一个长度为10的一维数组，而该函数需要提供的是一个正确答案的数字
    # 所以需要使用tf.argmax函数来得到正确答案对应的类别编号

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        y, tf.argmax(y_, 1))

    # 计算在当前batch中所有样例交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RAGE)
