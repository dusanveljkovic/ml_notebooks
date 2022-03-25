import tensorflow as tf
from utils import load_class_names, output_boxes, draw_outputs, resize_image
import cv2 as cv
import numpy as np
from yolov3 import YOLOv3Net

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
weight_file = 'weights/yolov3_weights.tf'
img_path = 'data/images/chech.jpg'

def main():
  
  model = YOLOv3Net(cfg_file, model_size, num_classes)
  model.load_weights(weight_file)
  
  class_names = load_class_names(class_name)
  
  image = cv.imread(img_path)
  image = np.array(image)
  image = tf.expand_dims(image, 0)
  
  resized_frame = resize_image(image, (model_size[0], model_size[1]))
  pred = model.predict(resized_frame)
  
  boxes, scores, classes, nums = output_boxes(\
    pred, model_size,
    max_output_size,
    max_output_size_per_class,
    iou_threshold,
    confidence_threshold)
  
  '''
  image = np.squeeze(image)
  img = draw_outputs(image, boxes, scores, classes, nums, class_names)
  
  
  win_name = 'Image detection'
  cv.imshow(win_name, img)
  cv.waitKey(0)
  cv.destroyAllWindows()
  '''
if __name__ == '__main__':
  main()