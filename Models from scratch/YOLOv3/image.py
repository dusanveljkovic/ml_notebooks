import tensorflow as tf
from utils import load_class_names, output_boxes, draw_outputs, resize_image
import cv2 as cv
import numpy as np
import os
from yolov3 import YOLOv3Net
from yolov33 import YoloV3

physical_devices = tf.config.experimental.list_logical_devices('GPU')
assert len(physical_devices) > 0, 'Not enough GPU hardware devices available'

model_size = (416, 416, 3)
num_classes = 80
class_name = './cfg/coco.names'
max_output_size = 40
max_output_size_per_class = 20
iou_threshold = 0.5
confidence_threshold = 0.5

cfg_file = './cfg/yolov3.cfg'
weight_file = 'weights/yolov3.tf'
img_path = 'data/images/koloseum.png'
model_path = 'saved_model/yolov3'


def main():
  
  '''
  if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
  else:
    model = YOLOv3Net(cfg_file, model_size, num_classes)
    model.load_weights(weight_file)
    
    model.save(model_path)
  '''
  
  model = YoloV3()
  model.load_weights(weight_file).expect_partial()
  
  class_names = load_class_names(class_name)
  
  img_raw = tf.image.decode_image(open(img_path, 'rb').read(), channels=3)

  img = tf.expand_dims(img_raw, 0)
  img = tf.image.resize(img, (416, 416))
  img = img / 255
  
  boxes, scores, classes, nums = model(img)
  
  print('detections:')
  for i in range(nums[0]):
    print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))
  
  img = cv.cvtColor(img_raw.numpy(), cv.COLOR_RGB2BGR)
  image = draw_outputs(img, (boxes, scores, classes, nums), class_names)
  
  
  win_name = 'Image detection'
  cv.imshow(win_name, image)
  cv.waitKey(0)
  cv.destroyAllWindows()
  
if __name__ == '__main__':
  main()