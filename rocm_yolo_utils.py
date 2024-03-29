
import cv2
import numpy as np
import ctypes

def generate_class_aggregation(labels):
  labels = np.array(labels)
  unique_labels = np.unique(labels)
  if len(unique_labels) == len(labels):
    # nothing to aggregate, so there is no mapping
    return None
  ret = []
  for label in unique_labels:
    if label == 'other':
      continue
    index = np.where(labels == label)[0]
    ret.append(((label, index[0]), index))
  return ret

def preprocess(src_image, detector_size = 640, keep_aspect_ratio = True):
  if isinstance(src_image, str):
    image = cv2.imread(src_image, cv2.IMREAD_COLOR)
  elif isinstance(src_image, np.ndarray):
    image = src_image
  else:
    raise ValueError("Invalid src_image")

  height, width, _ = image.shape

  downscaling_algorithm = cv2.INTER_LANCZOS4
  downscaling_algorithm = cv2.INTER_NEAREST
  downscaling_algorithm = cv2.INTER_AREA
  upscaling_algorithm = cv2.INTER_CUBIC

  if keep_aspect_ratio: # scale image to keep aspect ratio padding edges
    if width > height:
      scale_x = scale_y = width / detector_size
      image_resized = cv2.resize(image, (detector_size, int(round(height / scale_y))), downscaling_algorithm if scale_y > 1 else upscaling_algorithm)
      padding = ((0, image_resized.shape[1] - image_resized.shape[0]), (0, 0), (0, 0))
    else:
      scale_x = scale_y = height / detector_size
      image_resized = cv2.resize(image, (int(round(width / scale_x)), detector_size), downscaling_algorithm if scale_x > 1 else upscaling_algorithm)
      padding = ((0, 0), (0, image_resized.shape[0] - image_resized.shape[1]), (0, 0))
    image_resized = np.pad(image_resized, padding)
  else: # scale image to full detection area
    scale_x = width / detector_size
    scale_y = height / detector_size
    image_resized = cv2.resize(image, (detector_size, detector_size), downscaling_algorithm if scale_x > 1 else upscaling_algorithm)

  if True: # faster
    np_image = cv2.dnn.blobFromImage(image_resized, 1.0 / 255, (detector_size, detector_size), None, swapRB=True)
  else: # slower
    np_image = (1.0 / 255) * np.stack((image_resized[:,:,2], image_resized[:,:,1], image_resized[:,:,0]), dtype=np.float32)
    np_image = np.expand_dims(np_image, 0)

  return {'src_image': image, 'preprocessed_image': np_image, 'scale_x': scale_x, 'scale_y': scale_y}

def postprocess(preprocessed_data, detector_result, score_threshold = 0.25, nms_threshold = 0.4, vectorize = True, avoid_memory_copy = True, class_aggregation = None):
  if isinstance(detector_result, np.ndarray):
    npr = detector_result
  elif isinstance(detector_result, migraphx.argument):
    if avoid_memory_copy: # Migraphx offers a pointer to memory, use it to avoid memory copy
      addr = ctypes.cast(detector_result.data_ptr(), ctypes.POINTER(ctypes.c_float))
      npr = np.ctypeslib.as_array(addr, shape=detector_result.get_shape().lens())
    else: # Alternative in pure python:
      npr = np.ndarray(shape=detector_result.get_shape().lens(), buffer=np.array(detector_result.tolist()), dtype=float)
  else:
    raise ValueError(f'unknown detector_result: {type(detector_result)}')

  # Filter boxes
  boxes = []
  confidences = []
  class_ids = []

  model_box_count = npr.shape[2]
  model_class_count = npr.shape[1] - 4

  scale_x = preprocessed_data['scale_x']
  scale_y = preprocessed_data['scale_y']

  if vectorize: # fast numpy vectorized
    probs = npr[0, 4:, :].T
    if class_aggregation is not None:
      new_probs = np.zeros((probs.shape[0], len(class_aggregation)), dtype=probs.dtype)
      for index, ((label, class_id), selector) in enumerate(class_aggregation):
        new_probs[:, index] = np.sum(probs[:, selector], axis=1)
      probs = new_probs
    all_ids = np.argmax(probs, axis=1)
    all_confidences = probs[np.arange(model_box_count), all_ids]
    all_boxes = npr[0, 0:4, :].T
    mask = (all_confidences > score_threshold)
    class_ids = all_ids[mask]
    if class_aggregation is not None:
      class_ids = np.array([class_aggregation[index][0][1] for index in class_ids])
    confidences = all_confidences[mask]
    cx, cy, w, h = all_boxes[mask].T
    boxes = np.stack((scale_x * (cx - w / 2), scale_y * (cy - h / 2), scale_x * w, scale_y * h), axis=1)
  else: # slow, but readable
    assert class_aggregation is None, "not implemented"
    for i in range(0, model_box_count):
      row = npr[0, :, i]
      scores = row[4:]
      ids = np.argmax(scores)
      #ids = all_ids[i]
      confidence = scores[ids]

      if confidence > score_threshold:
        cx, cy, w, h = row[0:4]
        x = int(scale_x * (cx - w / 2))
        y = int(scale_y * (cy - h / 2))
        boxes.append([x, y, int(scale_x * w), int(scale_y * h)])
        confidences.append(float(confidence))
        class_ids.append(ids)

  indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)

  return [(boxes[index], class_ids[index], confidences[index]) for index in indexes]

def paint_boxes(src_image, boxes, labels = None):
  image = src_image.copy()
  font = cv2.FONT_HERSHEY_DUPLEX
  font_color = (255, 255, 255)
  border_thickness = 3

  for box, class_id, confidence in boxes:
    color = (0, 0, 255)
    x, y, w, h = box
    c0 = (int(round(x)), int(round(y)))
    c1 = (int(round(x + w)), int(round(y + h)))
    cv2.rectangle(image, c0, c1, color=color, thickness=border_thickness)

    #ty = y if y - font_size < 0 else y - font_size
    if labels is None or class_id >= len(labels):
      text = '%d %.3f' % (class_id, confidence)
    else:
      text = '%s %.3f' % (labels[class_id],  confidence)

    ((text_width, text_height), baseline) = cv2.getTextSize(text, font, 1, 1)

    rc0 = (c0[0] - border_thickness // 2, c0[1] + border_thickness // 2)
    text_height = text_height + baseline + 2
    if rc0[1] - text_height < 0:
      rc0 = (rc0[0], c0[1] + text_height - border_thickness // 2)
    cv2.rectangle(image, rc0, (rc0[0] + text_width, rc0[1] - text_height), color, -1)
    cv2.putText(image, text, (rc0[0], rc0[1] - baseline), font, 1, font_color)

  return image

if __name__ == '__main__':
  import migraphx
  import time
  import sys
  import argparse

  parser = argparse.ArgumentParser(description='Run yolov5+ detectors on an image', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('image', metavar='image', type=str, help='Path to image file')
  parser.add_argument('--labels', nargs='?', default='coco-labels.txt', type=str, help='Path to labels file')
  parser.add_argument('--output', nargs='?', default='output.jpg', type=str, help='Output image path')
  parser.add_argument('--model', nargs='?', default='yolov8n.mxr', type=str, help='MIGraphX model file')
  parser.add_argument('--benchmark', action='store_true', help='Benchmark inference')
  parser.add_argument('--quiet', action='store_true', help='Quiet operation')
  parser.add_argument('--nms-threshold', nargs='?', default=0.4, type=float, help='NMS threshold')
  parser.add_argument('--conserve-cpu', action='store_true', help='Use blocking mode in HIP synchronization conserving CPU')
  parser.add_argument('--aggregate-labels', action='store_true', help='Aggregate same-named labels in probabilities (discarding "other")')
  args = parser.parse_args();

  try:
    labels = [line.strip() for line in open(args.labels)]
  except:
    labels = None

  class_aggregation = None
  if args.aggregate_labels:
    assert labels is not None
    class_aggregation = generate_class_aggregation(labels)

  if args.conserve_cpu:
    ctypes.CDLL('/opt/rocm/lib/libamdhip64.so').hipSetDeviceFlags(4)

  model = migraphx.load(args.model)

  #print('get_parameter_names', model.get_parameter_names())
  #print('get_parameter_shapes', model.get_parameter_shapes())
  #print('get_output_shapes', model.get_output_shapes())

  model_input_name = model.get_parameter_names()[0];
  model_input_shape = model.get_parameter_shapes()[model_input_name];
  model_input_size = model_input_shape.lens()[-1]

  preprocess_t0 = time.time()
  image_data = preprocess(args.image, detector_size=model_input_size)
  preprocess_t1 = time.time()

  results = model.run({model_input_name: image_data['preprocessed_image']})

  if args.benchmark:
    N = 100
    inference_t0 = time.time()
    times = []
    for i in range(0, N):
      t0 = time.time()
      results = model.run({model_input_name: image_data['preprocessed_image']})
      t1 = time.time()
      times.append(t1 - t0)
    inference_t1 = time.time()
    inference_time = (inference_t1 - inference_t0) / N
    from statistics import median
    inference_time = median(times)

  postprocess_t0 = time.time()
  boxes = postprocess(image_data, results[0], nms_threshold = args.nms_threshold, class_aggregation = class_aggregation)
  postprocess_t1 = time.time()

  if args.benchmark:
    print('Inference time: %.4f s / %.1f fps (median), preprocess %.4f s, postprocess %.4f s' % (inference_time, 1 / inference_time, preprocess_t1 - preprocess_t0, postprocess_t1 - postprocess_t0))

  if not args.quiet:
    for box, class_id, confidence in boxes:
      label = class_id if labels is None or len(labels) <= class_id else labels[class_id]
      print(f'{box[0]:.1f} {box[1]:.1f} {box[0]+box[2]:.1f} {box[1]+box[3]:.1f}: {label} {confidence:.3f}')

  cv2.imwrite(args.output, paint_boxes(image_data['src_image'], boxes, labels))

  if not args.quiet:
    print(f'Written "{args.output}"')

