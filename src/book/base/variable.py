import tensorflow as tf

# 创建一个变量
state = tf.Variable(0, name="counter")
# 图
one = tf.constant(1)
plus = tf.add(state, one)
update = tf.assign(state, plus)
# 初始化
init_op = tf.global_variables_initializer()

# session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# 启动
sess.run(init_op)
sess.run(state)

for _ in range(3):
    # sess.run(update)
    print(sess.run([update, state]))
