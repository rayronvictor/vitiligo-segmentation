import gradio as gr
import PIL.Image as Image
from ultralytics import YOLO
from rembg import remove, new_session
import numpy as np
import cv2

# MODEL_PATH = "runs/segment/train3/weights/best.pt"
MODEL_PATH = "yolo11n-seg.pt"
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

def find_point_inside_sticker(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Lower red range
    lower_red1 = np.array([0, 120, 120])
    upper_red1 = np.array([5, 255, 255])
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)

    # Upper red range
    lower_red2 = np.array([175, 120, 120])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    final_mask = mask1 + mask2

    y_coords, x_coords = np.where(final_mask > 0)

    return final_mask, int(np.mean(x_coords)), int(np.mean(y_coords))


def calc_sticker_in_hand(input_image):
    """
    1. Upload an image of a hand with a reference sticker.
    2. Remove the image background, isolating the hand and sticker.
    3. Measure the pixel area of the hand (including the sticker).
    4. Find a point inside the sticker.
    5. Segment the sticker based on the point inside the sticker
    6. Measure the pixel area of the sticker.
    7. Return the segmented hand and sticker images and the sticker area in hand units.
    """

    # 2. Remove the image background, isolating the hand and sticker.
    hand_img_without_bg_pil = remove(input_image)
    hand_img_without_bg = np.array(hand_img_without_bg_pil)
    hand_img_without_bg_alpha = hand_img_without_bg[:, :, 3]

    # 3. Measure the pixel area of the hand (including the sticker).
    hand_num_pixels = np.count_nonzero(hand_img_without_bg_alpha)

    # 4. Find a point inside the sticker.
    final_mask, sticker_x, sticker_y = find_point_inside_sticker(hand_img_without_bg)

    print(f"Sticker point: ({sticker_x}, {sticker_y})")

    # 5. Segment the sticker based on the point inside the sticker
    session = new_session("sam")
    sam_prompt = [{"type": "point", "data": [sticker_x, sticker_y], "label": 1}]
    sticker_without_bg_pil = remove(input_image, session=session, sam_prompt=sam_prompt)
    sticker_without_bg = np.array(sticker_without_bg_pil)

    # # draw a test circle
    # cv2.circle(sticker_without_bg, (sticker_x, sticker_y), 7, (0, 255, 0), -1)  # Draw a green dot
    # cv2.putText(sticker_without_bg, f"({sticker_x}, {sticker_y})", (sticker_x - 40, sticker_y - 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 6. Measure the pixel area of the sticker.
    sticker_without_bg_alpha = sticker_without_bg[:, :, 3]
    sticker_num_pixels = np.count_nonzero(sticker_without_bg_alpha)

    return hand_img_without_bg, sticker_without_bg, sticker_num_pixels / hand_num_pixels

with gr.Blocks() as demo:
    gr.Label("Vitiligo Segmentation", container=False)

    with gr.Row():
        hand_img = gr.Image(type="pil", label="Upload a picture of your hand with sticker", height=400)
        # with gr.Group():
        segmented_hand_img = gr.Image(type="pil", label="Segmented hand", height=400)
        segmented_sticker_img = gr.Image(type="pil", label="Segmented sticker", height=400)
    with gr.Row():
        calibrate_btn= gr.Button("Calculate the sticker size", variant="primary")
        clear_btn = gr.Button("Clear")

    sticker_in_hands = gr.Textbox(label="Sticker size in hand units")

    clear_btn.click(
        fn=lambda: None,
        inputs=[],
        outputs=hand_img,
    )

    calibrate_btn.click(
        fn=calc_sticker_in_hand,
        inputs=hand_img,
        outputs=[segmented_hand_img, segmented_sticker_img, sticker_in_hands],
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
