from ultralytics import YOLO
from split_kfold import split_dataset
from pathlib import Path

MODEL_PATH = "yolo11n-seg.pt"
batch = 16
project = "vitiligo-poc"
epochs = 100
imgsz = 416

results = {}

dataset_path = Path("datasets/vitiligo-poc")
ds_yamls = split_dataset(dataset_path, ksplit=5)

for k, dataset_yaml in enumerate(ds_yamls):
    model = YOLO(MODEL_PATH)
    results[k] = model.train(
        data=dataset_yaml, imgsz=imgsz, epochs=epochs, batch=batch, project=project, name=f"fold_{k + 1}"
    )

# # Train the model
# results = model.train(data="datasets/vitiligo-poc/vitiligo-poc.yaml", epochs=10, imgsz=416)