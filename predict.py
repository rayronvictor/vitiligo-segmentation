from ultralytics import YOLO

from utils import plot_result

#model = YOLO("models/yolo11x-seg.pt")
model = YOLO("runs/segment/train2/weights/best.pt") # change to point to the trained weights

results = model("datasets/vitiligo-poc/test/images/5aeb0f6e-images--19-_jpg.rf.6f36d713f594b4172492fdba682cfcc9.jpg")  # predict on an image

plot_result(results[0])

# # Access the results
# for result in results:
#     xy = result.masks.xy  # mask in polygon format
#     xyn = result.masks.xyn  # normalized
#     masks = result.masks.data  # mask in matrix format (num_objects x H x W)