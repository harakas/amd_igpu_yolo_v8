
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

model.export(format="onnx", opset=12, imgsz=[640, 640])

