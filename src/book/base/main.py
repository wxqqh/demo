import tensorflow as tf

# 创建一个常量 op, 产生一个1*2的矩阵. 这个op被作为一个节点加到默认图中
# 构造器返回值代表该常量 op 的返回值
matrix1 = tf.constant([[3., 3.]])

# 创建另外一个常量 op, 产生一个 2*1 的矩阵
matrix2 = tf.constant([[1.], [2.]])

# 创建一个矩阵乘法 matmul op
product = tf.matmul(matrix1, matrix2)

# 启动默认视图, 打印运行设备
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# 调用sess.run方法来执行矩阵乘法op, 传入 product 参数
# 整个过程是自动化, session负责传递 op 所需要的全部输入, op 通常是并发执行
result = sess.run(product)

print(result)

# 关闭会话
sess.close()
