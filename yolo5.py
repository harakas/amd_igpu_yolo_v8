# An example script of running ultralytics/yolov5 inference on an image

###
# Load image
from PIL import Image
input_image = Image.open('wolf.jpg')

imgs = [
    input_image
]


###
# Import torch and print debug about GPU
import torch

print('torch.cuda.is_available()', torch.cuda.is_available())
print('torch.cuda.device_count()', torch.cuda.device_count())
print('torch.cuda.current_device()', torch.cuda.current_device())
print('torch.cuda.get_device_name(0)', torch.cuda.get_device_name(0))


###
# Load/download the model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)


###
# Run inference
results = model(imgs)

# Save results
results.print()
results.save()


###
# Small benchmark
import time
from statistics import median

timings = []
for i in range(0,10):
  t0 = time.time()
  results2 = model(imgs)
  t1 = time.time()
  timings.append(t1 - t0)

print(f'Benchmark: inference min {min(timings)}, median {median(timings)}')

