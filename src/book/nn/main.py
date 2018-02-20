import tensorflow as tf
from numpy.random import RandomState

# 定义训练数据batch大小
batch_saze = 8
# 定义神经网络的参数,
with tf.name_scope("Weight"):
    w1 = tf.Variable(tf.random_normal(
        [2, 3], stddev=1,  seed=1), name="w1")
    w2 = tf.Variable(tf.random_normal(
        [3, 1], stddev=1,  seed=1), name="w2")

# 在shape的一个维度上使用None可以方便使用不同的batch大小
# 在训练时需要把数据分成较小的batch, 但是在测试时候, 可以一次性使用全部的数据
# 当数据集比较小的时候方便测试, 但是若相反, 把大量的数据放入一个batch中可能会导致内存溢出
with tf.name_scope("Input"):
    x = tf.placeholder(tf.float32, [None, 2], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, 1], name="y-input")

# 定义神经网络传播过程
with tf.name_scope("Hidden-Layer"):
    a = tf.matmul(x, w1, name="hidden-layer")
with tf.name_scope("Output-Layer"):
    y = tf.matmul(a, w2, name="output-layer")

# 定义损失函数和反向传播的算法
with tf.name_scope("Loss-FN"):
    y1 = tf.sigmoid(y)
    cross_entropy = -tf.reduce_mean(
        y_ * tf.log(tf.clip_by_value(y1, 1e-10, 1.0))
        + (1-y1) * tf.log(tf.clip_by_value(1-y1, 1e-10, 1.0))
    )

with tf.name_scope("Train"):
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成一个模拟的数据集
rdm = RandomState()
dataset_size = 128

X = rdm.rand(dataset_size, 2)

# 定义规则来给出样本的标签, 在这里所有的x1+x2<1的样例都认为是正样本(零件合格)
# 而其他为负样本
# 和Tensorflow playground 中表示不一样的地方是:
#   这里用0表示负样本, 1表示正样本
# 大部分解决分类问题的神经网络都会使用 0 和 1 的表示方法

Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

saver = tf.train.Saver()

# 创建一个session来运行

with tf.Session(
        config=tf.ConfigProto(allow_soft_placement=True
                              # ,log_device_placement=True
                              )) as sess:
    init = tf.global_variables_initializer()

    # 初始化变量
    sess.run(init)
    # 在训练之前神经网络的值
    print("------ before train-----")
    print("w1 : ", sess.run(w1), "  w2:  ", sess.run(w2))
    STEPS = 25000
    print("------ start train-----")

    # 声明tensorboard log的输出对象
    writer = tf.summary.FileWriter("./log", graph=tf.get_default_graph())

    for i in range(STEPS):
        # 每次选取batch_size个样本进行训练
        start = (i * batch_saze) % dataset_size
        end = min(start + batch_saze, dataset_size)

        if i % 1000 == 0:
            # 配置运行时需要记录的信息
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # 运行时记录信息的proto
            run_metadata = tf.RunMetadata()
            # 通过选取的样本训练神经网络并且更新参数
            # 将配置信息和记录运行信息的proto传入运行的过程, 从而记录运行时的每一个节点的时间和空间开销的信息
            sess.run(train_step, feed_dict={
                x: X[start:end],
                y_: Y[start:end]
            }, options=run_options, run_metadata=run_metadata)

            writer.add_run_metadata(run_metadata, "step %03d" % i)

            total_coress_entropy = sess.run(
                cross_entropy, feed_dict={x: X, y_: Y})
            print("after train step: %f, cross_entropy is: %f" %
                  (i, total_coress_entropy))
            print("w1 : ", sess.run(w1), "  w2:  ", sess.run(w2))
        else:
            sess.run(train_step, feed_dict={
                x: X[start:end],
                y_: Y[start:end]
            })
    print("------ end train-----")
    saver.save(sess, "./model/train")
    writer.close()
