# An example script of running ultralytics/yolov8 inference on an image

###
# Load image
from PIL import Image
input_image = Image.open('wolf.jpg')

imgs = [
    input_image
]

###
# Download and load model
# See https://github.com/ultralytics/ultralytics and https://docs.ultralytics.com/datasets/detect/open-images-v7/ for available models
import urllib.request
urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-oiv7.pt', 'yolov8n-oiv7.pt')

from ultralytics import YOLO
model = YOLO('yolov8n-oiv7.pt')

###
# Run inference
model.predict(imgs, save=True)

