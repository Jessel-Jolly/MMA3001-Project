from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/pork_rasher_detection/weights/best.pt")

# Run evaluation on test split
metrics = model.val(data="Pork Rasher Error-Packaging-.v4i.yolov11/data.yaml", split="test")

# Access per-class mAP and confusion matrix
print("Per-class mAP50:", metrics.box.ap50)
print("Confusion Matrix saved to:", metrics.save_dir)