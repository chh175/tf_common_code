import tensorflow as tf
import vgg

from tensorflow.python.framework import graph_util


slim = tf.contrib.slim

g = tf.Graph()
with g.as_default():
    inputs = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input')
    with slim.arg_scope(vgg.vgg_arg_scope()):
        outputs, end_points = vgg.vgg_16(inputs,
                                         is_training=False)
    

    outputs = tf.nn.softmax(outputs, name='output')
    
    saver = tf.train.Saver()
    with tf.Session(graph = g) as sess:
        saver.restore(sess, 'vgg_16.ckpt')
        
        output_graph_def = graph_util.convert_variables_to_constants(
                sess, g.as_graph_def(), output_node_names=['output']) 
        
        with tf.gfile.FastGFile('my_test.pb', mode = 'wb') as f:
            f.write(output_graph_def.SerializeToString())