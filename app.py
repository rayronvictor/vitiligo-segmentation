import gradio as gr
import PIL.Image as Image
from ultralytics import YOLO

MODEL_PATH = "runs/segment/train3/weights/best.pt"
model = YOLO(MODEL_PATH)

def predict_image(img, conf_threshold, iou_threshold):
    """Predicts objects in an image using a YOLO11 model with adjustable confidence and IOU thresholds."""
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=416,
    )

    for r in results:
        im_array = r.plot(boxes=False)
        im = Image.fromarray(im_array[..., ::-1])

    return im

iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.7, label="IoU threshold"),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="Vitiligo Segmentation",
    description="Upload images for inference.",
)

if __name__ == "__main__":
    iface.launch()
