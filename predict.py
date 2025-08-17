from ultralytics import YOLO

from utils import plot_result

#model = YOLO("models/yolo11x-seg.pt")
model = YOLO("runs/segment/train/weights/best.pt") # change to point to the trained weights

results = model("datasets/vitiligo-poc/test/images/e3b95dde-roi306_jpg.rf.5ce06976f42b0c32bff3948336dbd72f.jpg")  # predict on an image

plot_result(results[0])

# # Access the results
# for result in results:
#     xy = result.masks.xy  # mask in polygon format
#     xyn = result.masks.xyn  # normalized
#     masks = result.masks.data  # mask in matrix format (num_objects x H x W)