
import time
import cv2
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import migraphx
import ctypes

labels = [line.strip() for line in open('coco-labels.txt')]

image = Image.open('wolf.jpg')

width, height = image.size

if True: # scale image to keep aspect ratio padding edges
  if width > height:
    scale_x = scale_y = width / 640.0
    image_resized = image.resize((640, int(round(height / scale_y))), Image.LANCZOS if scale_y > 1 else Image.BICUBIC)
  else:
    scale_x = scale_y = height / 640.0
    image_resized = image.resize((int(round(width / scale_x)), 640), Image.LANCZOS if scale_x > 1 else Image.BICUBIC)

  image_resized = ImageOps.pad(image_resized, (640, 640), centering=(0, 0))
else: # scale image to full detection area
  scale_x = width / 640.0
  scale_y = height / 640.0
  image_resized = image.resize((640, 640), Image.LANCZOS)

# Rearrange image channels, convert to float32 and normalize to [0..1]
np_image = np.asarray(image_resized)
# PIL already uses proper RGB order
np_image = (1.0 / 255) * np.stack((np_image[:,:,0], np_image[:,:,1], np_image[:,:,2]), dtype=np.float32)
#np_image = (1.0 / 255) * np.stack((np_image[:,:,2], np_image[:,:,1], np_image[:,:,0]), dtype=np.float32)
np_image = np.expand_dims(np_image, 0)

model = migraphx.load("yolov8n.mxr")

#print('get_parameter_names', model.get_parameter_names())
#print('get_parameter_shapes', model.get_parameter_shapes())
#print('get_output_shapes', model.get_output_shapes())

input_name = next(iter(model.get_parameter_shapes()))
input_argument = migraphx.argument(np_image)

results = model.run({input_name: np_image})

N = 10
t0 = time.time()
for i in range(0, N):
  results = model.run({input_name: np_image})
t1 = time.time()
print('Inference time: %.3f s / %.1f fps' % ((t1 - t0) / N, N / (t1 - t0)))

if True: # Migraphx offers a pointer to memory, use it to avoid memory copy
  addr = ctypes.cast(results[0].data_ptr(), ctypes.POINTER(ctypes.c_float))
  npr = np.ctypeslib.as_array(addr, shape=results[0].get_shape().lens())
else: # Alternative in pure python:
  npr = np.ndarray(shape=results[0].get_shape().lens(), buffer=np.array(results[0].tolist()), dtype=float)

# Filter boxes
boxes = []
confidences = []
class_ids = []

model_box_count = npr.shape[2]
model_class_count = npr.shape[1] - 4

if True: # fast numpy vectorized
  probs = npr[0, 4:, :]
  all_ids = np.argmax(probs, axis=0)
  all_confidences = np.take(probs.T, model_class_count*np.arange(0, model_box_count) + all_ids)
  all_boxes = npr[0, 0:4, :].T
  mask = (all_confidences > 0.25)
  class_ids = all_ids[mask]
  confidences = all_confidences[mask]
  cx, cy, w, h = all_boxes[mask].T
  boxes = np.stack((scale_x * (cx - w / 2), scale_y * (cy - h / 2), scale_x * w, scale_y * h), axis=1)
else: # slow, but readable
  for i in range(0, model_box_count):
    row = npr[0, :, i]
    scores = row[4:]
    ids = np.argmax(scores)
    #ids = all_ids[i]
    confidence = scores[ids]

    if confidence > 0.25:
      cx, cy, w, h = row[0:4]
      x = int(scale_x * (cx - w / 2))
      y = int(scale_y * (cy - h / 2))
      boxes.append([x, y, int(scale_x * w), int(scale_y * h)])
      confidences.append(float(confidence))
      class_ids.append(ids)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.4)

# Print result
for index in indexes:
  print(boxes[index], confidences[index], labels[class_ids[index]])

# Draw boxes on the image
alpha_image = image.convert('RGBA')
draw = ImageDraw.Draw(alpha_image)

font_size = 30
font = ImageFont.truetype("Pillow/Tests/fonts/FreeMonoBold.ttf", font_size)

for index in indexes:
  x, y, w, h = boxes[index]
  draw.line([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)], fill="red", width=3)
  ty = y if y - font_size < 0 else y - font_size
  #text = '%d %.3f' % (class_ids[index],  confidences[index])
  text = '%s %.3f' % (labels[class_ids[index]],  confidences[index])
  draw.rectangle([x, ty, x + font.getlength(text), ty + font_size], fill="red")
  draw.text([x, ty], text, fill="white", font=font)

alpha_image.convert('RGB').save('output.jpg', 'JPEG', quality=85)

print('Written "output.jpg"')

