import tensorflow as tf



graph_def_file = "mobilenet_v2_float_flower.pb"
input_arrays = ["input"]
output_arrays = ["output"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("mobilenet_v2_float_flower.tflite", "wb").write(tflite_model)