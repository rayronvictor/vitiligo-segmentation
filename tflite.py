from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n-seg.pt")

# Export the model to TFLite format
# model.export(format="tflite", nms=True)  # creates 'yolo11n_float32.tflite'
model.export(format="tflite", nms=True)  # creates 'yolo11n_float32.tflite'

# # Load the exported TFLite model
# tflite_model = YOLO("yolo11n_float32.tflite")
#
# # Run inference
# results = tflite_model("https://ultralytics.com/images/bus.jpg")