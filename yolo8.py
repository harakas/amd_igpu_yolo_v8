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

###
# Small benchmark
imgs = [input_image] * 2
import time
from statistics import median

print('Benchmarking..')
timings = []
for i in range(0, 100):
  t0 = time.time()
  results2 = model.predict(imgs, verbose=False)
  t1 = time.time()
  timings.append((t1 - t0) / len(imgs))
  if len(timings) > 10:
    print(f'{i} inference last {t1 - t0:.4f}, min {min(timings)}, median {median(timings)}')

