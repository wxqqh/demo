import tensorflow as tf

v1 = tf.Variable(tf.random_normal([1, 2]), name="v1")
v2 = tf.Variable(tf.random_normal([1, 2]), name="v2")

result = tf.add(v1, v2, name="result")

for variables in tf.global_variables():
    print(variables.name)

saver = tf.train.Saver()

# 通过tf.train.Saver类实现保存和载入神经网络模型
# 这里会生成
# checkpoint
#   checkpoint文件保存了一个目录下所有的模型文件列表，这个文件是tf.train.Saver类自动生成且自动维护的。
#   在 checkpoint文件中维护了由一个tf.train.Saver类持久化的所有TensorFlow模型文件的文件名。
#   当某个保存的TensorFlow模型文件被删除时，这个模型所对应的文件名也会从checkpoint文件中删除。
#   checkpoint中内容的格式为CheckpointState Protocol Buffer.
# test.ckpt.data-xxxxx-of-xxxxx
#   当次训练的所有的数据
# test.ckpt.index
#   件保存了TensorFlow程序中每一个变量的取值，这个文件是通过SSTable格式存储的，可以大致理解为就是一个（key，value）列表。
#   model.ckpt文件中列表的第一行描述了文件的元信息，比如在这个文件中存储的变量列表。
#   列表剩下的每一行保存了一个变量的片段，变量片段的信息是通过SavedSlice Protocol Buffer定义的。
#   SavedSlice类型中保存了变量的名称、当前片段的信息以及变量取值。
#   TensorFlow提供了tf.train.NewCheckpointReader类来查看model.ckpt文件中保存的变量信息。
# test.ckpt.meta
#   model.ckpt.meta文件保存了TensorFlow计算图的结构
#   所有的 variables, operations, collections等等, 可以理解为神经网络的网络结构./
#   TensorFlow通过元图（MetaGraph）来记录计算图中节点的信息以及运行计算图中节点所需要的元数据。
#   TensorFlow中元图是由MetaGraphDef Protocol Buffer定义的。
#   MetaGraphDef 中的内容构成了TensorFlow持久化时的第一个文件。
#   保存MetaGraphDef 信息的文件默认以.meta为后缀名，文件model.ckpt.meta中存储的就是元图数据。
#

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("v1:", sess.run(v1))  # 打印v1、v2的值和之前的进行对比
    print("v2:", sess.run(v2))
    print("result:", sess.run(result))
    saver.save(sess, "./Model/test.ckpt")
