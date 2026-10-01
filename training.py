from pathlib import Path
from ultralytics import YOLO

# Path to your dataset's data.yaml file
yaml_path = Path( "Pork Rasher Error-Packaging-.v4i.yolov11/data.yaml" )
# Load pre-trained YOLOv11 nano model
model = YOLO("yolo11n.pt")

# Train the model
results = model.train(data=str(yaml_path), epochs=50, imgsz=640, batch=16,
    name="pork_rasher_detection")