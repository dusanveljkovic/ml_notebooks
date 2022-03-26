import enum
import tensorflow as tf
import numpy as np
import cv2 as cv

def non_max_suppression(inputs, model_size, max_output_size,
                        max_output_size_per_class, iou_threshold, confidence_threshold):
  bbox, confs, class_probs = tf.split(inputs, [4, 1, -1], axis=-1)
  bbox /= model_size[0]
  
  scores = confs * class_probs
  boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
    boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
    scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
    max_output_size_per_class=max_output_size_per_class,
    max_total_size=max_output_size,
    iou_threshold=iou_threshold,
    score_threshold=confidence_threshold
  )
  
  return boxes, scores, classes, valid_detections

def resize_image(inputs, modelsize):
  inputs = tf.image.resize(inputs, modelsize)
  return inputs

def load_class_names(file_name):
  with open(file_name, 'r') as f:
    class_names = f.read().splitlines()
  return class_names

def output_boxes(inputs, model_size, max_output_size, max_output_size_per_class,
                 iou_threshold, confidence_threshold):
  
  center_x, center_y, width, height, confidence, classes = \
    tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)
    
  top_left_x = center_x - width / 2.0
  top_left_y = center_y - height / 2.0
  bottom_right_x = center_x + width / 2.0
  bottom_right_y = center_y + height / 2.0
  
  inputs = tf.concat([top_left_x, top_left_y, bottom_right_x,
                      bottom_right_y, confidence, classes], axis=-1)
  
  boxes_dicts = non_max_suppression(inputs, model_size, max_output_size,
                                    max_output_size_per_class, iou_threshold, confidence_threshold)
  
  return boxes_dicts

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

def load_darknet_weights(model, weight_file):
  wf = open(weight_file, 'rb')
  major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
  
  layers = YOLOV3_LAYER_LIST
  
  for layer_name in layers:
    sub_model = model.get_layer(layer_name)
    for i, layer in enumerate(sub_model.layers):
      if not layer.name.startswith('conv2d'):
        continue
      batch_norm = None
      if i + 1 < len(sub_model.layers) and sub_model.layers[i + 1].name.startswith('batch_norm'):
        batch_norm = sub_model.layers[i + 1]
          
      print('{}/{} {}'.format(
        sub_model.name, layer.name, 'bn' if batch_norm else 'bias'
      ))
      
      filters = layer.filters
      size = layer.kernel_size[0]
      in_dim = layer.get_input_shape_at(0)[-1]
      
      if batch_norm is None:
        conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
      else:
        bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
        
        bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
      
      conv_shape = (filters, in_dim, size, size)
      conv_weights = np.fromfile(
        wf, dtype=np.float32, count=np.product(conv_shape) 
      )
      conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])
      
      if batch_norm is None:
        layer.set_weights([conv_weights, conv_bias])
      else:
        layer.set_weights([conv_weights])
        batch_norm.set_weights(bn_weights)
  assert len(wf.read()) == 0, 'failed to read all data'
  wf.close()
  
def draw_outputs(img, outputs, class_names):
  boxes, objectness, classes, nums = outputs
  boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
  boxes = np.array(boxes)
  wh = np.flip(img.shape[0:2])
  for i in range(nums):
    x1y1 = tuple((boxes[i,0:2] * [img.shape[1],img.shape[0]]).astype(np.int32))
    x2y2 = tuple((boxes[i,2:4] * [img.shape[1],img.shape[0]]).astype(np.int32))
    img = cv.rectangle(img, (x1y1), (x2y2), (255, 0, 0), 2)
    img = cv.putText(img, '{} {:.4f}'.format(
      class_names[int(classes[i])], objectness[i]),
      x1y1, cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
  return img
  