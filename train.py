from ultralytics import YOLO

# Load a model
model = YOLO("yolo11x-seg.pt")

# Train the model
results = model.train(data="datasets/vitiligo-poc/vitiligo-poc.yaml", epochs=10, imgsz=416)
