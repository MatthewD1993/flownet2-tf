from tensorflow.contrib.framework.python.framework.checkpoint_utils import list_variables
import tensorflow as tf
result = list_variables("./checkpoints/FlowNetS/flownet-S.ckpt-0")
for name, shape in result:
    print(name, shape)

print("Finished!")

# saver = tf.train.Saver()
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./checkpoints/FlowNetS/flownet-S.ckpt-0.meta')
    saver.restore(sess, "./checkpoints/FlowNetS/flownet-S.ckpt-0")
    graph = tf.get_default_graph()
    step = graph.get_tensor_by_name('global_step:0')
    print(sess.run(step))
