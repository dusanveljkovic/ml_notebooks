from yolov33 import YoloV3
from utils import load_darknet_weights
import numpy as np

weights_file = './cfg/yolov3.weights'
output_path = './weights/yolov3.tf'

def main():
  yolo = YoloV3(classes=80)
  
  yolo.summary()
  
  load_darknet_weights(yolo, weights_file)
  
  img = np.random.random((1, 320, 320, 3)).astype(np.float32)
  output = yolo(img)
  
  yolo.save_weights(output_path)
  
if __name__ == '__main__':
  main()