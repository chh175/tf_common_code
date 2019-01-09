import math
import tensorflow as tf

import align_lfw
import matplotlib.pyplot as plt

slim = tf.contrib.slim
batch_size = 32

dataset_dir = 'H:\\align_lfw_face_tfrecord'
TFrecord_dataset = align_lfw



def dataset_input(is_training):
    
    if is_training:
        dataset = TFrecord_dataset.get_dataset('train', dataset_dir)                                          
    else:
        dataset = TFrecord_dataset.get_dataset('validation', dataset_dir)
                                          

    provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      shuffle=is_training,
      common_queue_capacity=2 * batch_size,
      common_queue_min=batch_size)
    [image, label] = provider.get(['image', 'label'])
    
    image.set_shape((112, 96, 3))

    if is_training:
        image = tf.image.random_flip_left_right(image)

   
    images, labels = tf.train.batch(
      tensors=[image, label],
      batch_size=batch_size,
      num_threads=4,
      capacity=5 * batch_size)
    
    return images, labels


images, labels = dataset_input(True)

sess = tf.Session()
coord = tf.train.Coordinator()
tf.train.start_queue_runners(coord=coord, sess=sess)

my_images, my_labels = sess.run([images, labels])

plt.imshow(my_images[5]) 
   