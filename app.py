import gradio as gr
import PIL.Image as Image
from ultralytics import YOLO
from rembg import remove
import numpy as np
import cv2

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

def calc_sticker_in_hand(input_image):
    """
    1. Upload an image of a hand with a reference sticker.
    2. Remove the image background, isolating the hand and sticker.
    3. Measure the pixel area of the hand (including the sticker).
    4. Measure the pixel area of the sticker.
    5. Return the sticker/hand ratio by dividing the sticker's area by the hand's area.
    """

    # 2. Remove background from the image
    img_without_bg_pil = remove(input_image)
    img_without_bg = np.array(img_without_bg_pil)

    # 3. Measure the pixel area of the hand (including the sticker)
    img_without_bg_alpha = img_without_bg[:, :, 3]
    hand_num_pixels = np.count_nonzero(img_without_bg_alpha)

    # 4. Measure the pixel area of the sticker.
    # --- Red Pixel Filtering using HSV ---
    # Convert the RGBA image to RGB (necessary for HSV conversion)
    rgb_image = cv2.cvtColor(img_without_bg, cv2.COLOR_RGBA2RGB)

    # Convert the RGB image to HSV
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Define the HSV range for the color red. Red wraps around the 0-180 hue scale in OpenCV.
    # Lower range (covers bright reds)
    lower_red = np.array([0, 70, 100])
    upper_red = np.array([5, 255, 255])
    mask1 = cv2.inRange(hsv_image, lower_red, upper_red)

    # Upper range (covers darker, more saturated reds)
    lower_red2 = np.array([175, 70, 100])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    # Combine the two masks to get all red pixels
    red_hue_mask = mask1 + mask2

    # Final mask: The pixel must be red AND part of the foreground
    final_red_mask = cv2.bitwise_and(red_hue_mask, red_hue_mask, mask=img_without_bg_alpha)

    # Count the number of red pixels in the final mask
    red_filtered_pixels = np.count_nonzero(final_red_mask)

    return final_red_mask, hand_num_pixels

with gr.Blocks() as demo:
    gr.Label("Vitiligo Segmentation", container=False)

    with gr.Row():
        hand_img = gr.Image(type="pil", label="Upload a picture of your hand with sticker")
        result_img = gr.Image(type="pil", label="Result")
    with gr.Row():
        calibrate_btn= gr.Button("Calculate the Sticker/Hand Ratio", variant="primary")
        clear_btn = gr.Button("Clear")

    sticker_hand_ratio = gr.Textbox(label="Sticker/Hand Ratio")

    clear_btn.click(
        fn=lambda: None,
        inputs=[],
        outputs=hand_img,
    )

    calibrate_btn.click(
        fn=calc_sticker_in_hand,
        inputs=hand_img,
        outputs=[result_img, sticker_hand_ratio],
    )

    # with gr.Row():
    #     gr.Image(type="pil", label="Upload a picture of your hand with sticker")
    #     gr.Image(type="pil", label="Result")
    # with gr.Row():
    #     with gr.Accordion("Advanced options", open=False):
    #         gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold")
    #         gr.Slider(minimum=0, maximum=1, value=0.7, label="IoU threshold")
    # with gr.Row():
    #     gr.Button("Clear")
    #     gr.Button("Predict", variant="primary")


    # with gr.Row():
    #     with gr.Column():
    #         gr.Image(type="pil", label="Upload Image")
    #
    #         with gr.Accordion("Advanced options", open=False):
    #             gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
    #             gr.Slider(minimum=0, maximum=1, value=0.7, label="IoU threshold"),
    #
    #
    #     gr.Image(type="pil", label="Result")

# demo = gr.Interface(
#     fn=predict_image,
#     inputs=[
#         gr.Image(type="pil", label="Upload Image"),
#     ],
#     additional_inputs=[
#         gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
#         gr.Slider(minimum=0, maximum=1, value=0.7, label="IoU threshold"),
#     ],
#     additional_inputs_accordion="Advanced options",
#     outputs=gr.Image(type="pil", label="Result"),
#     title="Vitiligo Segmentation",
#     description="Upload images for inference.",
# )

if __name__ == "__main__":
    demo.launch()
