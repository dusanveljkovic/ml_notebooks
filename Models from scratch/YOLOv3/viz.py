import gradio as gr
from yolov3 import YoloV3
from utils import load_class_names, draw_outputs
import cv2 as cv
import tensorflow as tf

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

def process_image(model, class_names):
  def f(img_raw):
    img = tf.expand_dims(img_raw, 0)
    img = tf.image.resize(img, (416, 416))
    img = img / 255
    
    boxes, scores, classes, nums = model(img)
    
    image = draw_outputs(img_raw, (boxes, scores, classes, nums), class_names)
  
    return image
  return f

def dummy(vid_path):
  print(vid_path)
  return '/tmp/TMP_OUT_714939653348843.mp4'

def main():
  model = YoloV3()
  model.load_weights(weight_file).expect_partial()
  
  class_names = load_class_names(class_name)
  
  func = process_image(model, class_names)
  
  iface = gr.Interface(fn=func, inputs=gr.inputs.Image(), outputs=gr.outputs.Image())

  iface.launch()
  
  
if __name__ == '__main__':
  main()
