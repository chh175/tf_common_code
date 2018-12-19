import tensorflow as tf

from tensorflow.python.framework import graph_util

from nets import nets_factory

slim = tf.contrib.slim


model_name = 'mobilenet_v2'
num_classes = 102
is_training = False
image_size = 224
ckpt_name = 'model.ckpt-10000'
freeze_pb_name = 'mobilenet_v2_float_flower.pb'



g = tf.Graph()
with g.as_default():
    
    network_fn = nets_factory.get_network_fn(
                                            model_name,
                                            num_classes=num_classes,
                                            is_training=is_training)
    
    placeholder = tf.placeholder(name='input', dtype=tf.float32,
                                 shape=[None, image_size, image_size, 3])

    
    outputs, end_points = network_fn(placeholder)
    outputs = tf.nn.softmax(outputs, name='output')
    saver = tf.train.Saver()
    
    
    with tf.Session(graph = g) as sess:
        saver.restore(sess, ckpt_name)
        
        output_graph_def = graph_util.convert_variables_to_constants(
                sess, g.as_graph_def(), output_node_names=['output']) 
        
        with tf.gfile.FastGFile(freeze_pb_name, mode = 'wb') as f:
            f.write(output_graph_def.SerializeToString())