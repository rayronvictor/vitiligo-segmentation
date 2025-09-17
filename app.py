import gradio as gr
import PIL.Image as Image
from ultralytics import YOLO
from rembg import remove, new_session
import numpy as np
import cv2

MODEL_PATH = "models/vitiligon-seg.pt"
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

    # for r in results:
    #     print(r.masks.data.shape)
    #     im_array = r.plot(boxes=False)
    #     im = Image.fromarray(im_array[..., ::-1])

    r = results[0]

    mask = r.masks.data.sum(dim=0)
    mask_array = (mask.cpu().numpy() * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_array)

    im_array = r.plot(boxes=False)
    im = Image.fromarray(im_array[..., ::-1])

    return im, mask_pil


def find_point_inside_sticker(image, color_ranges):
    if isinstance(image, Image.Image):
        image = np.asarray(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    final_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for color_range in color_ranges:
        lower_color = np.array(color_range[0])  # [0, 120, 120]
        upper_color = np.array(color_range[1])  # [5, 255, 255]
        mask = cv2.inRange(hsv_image, lower_color, upper_color)

        final_mask = final_mask + mask

    y_coords, x_coords = np.where(final_mask > 0)

    if len(y_coords) > 0 or len(x_coords) > 0:
        return final_mask, int(np.mean(x_coords)), int(np.mean(y_coords))
    else:
        return final_mask, 0, 0

def segment_from_point(input_image, x, y):
    session = new_session("sam")
    sam_prompt = [{"type": "point", "data": [x, y], "label": 1}]
    image_without_bg_pil = remove(input_image, session=session, sam_prompt=sam_prompt)
    return np.array(image_without_bg_pil)

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
    final_mask, sticker_x, sticker_y = find_point_inside_sticker(
        hand_img_without_bg,
        [
            # Blue range
            [[100, 80, 80],
             [140, 200, 200]],
        ]
        # [
        #     # Red range
        #     [[0, 120, 120], [5, 255, 255]],
        #     # Upper red range
        #     [[175, 120, 120], [180, 255, 255]]
        # ],

    )

    print(f"Sticker point: ({sticker_x}, {sticker_y})")

    # 5. Segment the sticker based on the point inside the sticker
    sticker_without_bg = segment_from_point(input_image, sticker_x, sticker_y)

    # # draw a test circle
    # cv2.circle(sticker_without_bg, (sticker_x, sticker_y), 7, (0, 255, 0), -1)  # Draw a green dot
    # cv2.putText(sticker_without_bg, f"({sticker_x}, {sticker_y})", (sticker_x - 40, sticker_y - 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 6. Measure the pixel area of the sticker.
    sticker_without_bg_alpha = sticker_without_bg[:, :, 3]
    sticker_num_pixels = np.count_nonzero(sticker_without_bg_alpha)

    return hand_img_without_bg, sticker_without_bg, sticker_num_pixels / hand_num_pixels

def calc_vitiligo_area_in_hand(input_image, sticker_in_hands_unit, conf_threshold, iou_threshold):
    """
    1. Find a point inside the sticker.
    2. Segment the sticker based on the point inside the sticker.
    3. Measure the pixel area of the sticker.
    4. Calculate the pixel size in hand units.
    5. Segment the vitiligo area.
    6. Measure the vitiligo area in hand units.
    """

    # 1. Find a point inside the sticker.
    final_mask, sticker_x, sticker_y = find_point_inside_sticker(
        np.array(input_image),
        [
            # Blue range
            [[100, 80, 80],
             [140, 200, 200]],
        ]
    )

    print(f"Sticker point: ({sticker_x}, {sticker_y})")

    # 2. Segment the sticker based on the point inside the sticker
    sticker_without_bg = segment_from_point(input_image, sticker_x, sticker_y)

    # 3. Measure the pixel area of the sticker.
    sticker_without_bg_alpha = sticker_without_bg[:, :, 3]
    sticker_num_pixels = np.count_nonzero(sticker_without_bg_alpha)

    # 4. Calculate the pixel size in hand units
    pixel_size_in_hands = float(sticker_in_hands_unit) / sticker_num_pixels

    # 5. Segment the vitiligo area
    im, mask = predict_image(input_image, conf_threshold, iou_threshold)

    vitiligo_num_pixels = np.count_nonzero(mask)

    # 6. Measure the vitiligo area in hand units
    return im, mask, vitiligo_num_pixels * pixel_size_in_hands

debug = True

with gr.Blocks() as demo:
    gr.Label("Vitiligo Segmentation", container=False)

    with gr.Row():
        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                # Step 1
                Upload an image of a hand with a reference sticker.
                """
            )
            hand_img = gr.Image(type="pil", label="Upload an image", height=400)
            calibrate_btn = gr.Button("Calculate the sticker size", variant="primary")
            # clear_btn = gr.Button("Clear")
            sticker_in_hands = gr.Textbox(label="Sticker size in hand units", visible=debug)
            with gr.Row():
                segmented_hand_img = gr.Image(type="pil", label="Segmented hand", height=400, visible=debug)
                segmented_sticker_img = gr.Image(type="pil", label="Segmented sticker", height=400, visible=debug)

        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                # Step 2
                Upload an image of a anatomical area with vitiligo and a reference sticker.
                """
            )
            area_img = gr.Image(type="pil", label="Upload a picture of your anatomical area with sticker", height=400)
            calculate_btn = gr.Button("Calculate the vitiligo area", variant="primary")
            with gr.Accordion("Advanced options", open=False):
                confidence = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold")
                iou = gr.Slider(minimum=0, maximum=1, value=0.7, label="IoU threshold")
            segmented_area_sticker_img = gr.Image(type="pil", label="Segmented sticker", height=400, visible=debug)
        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                # Result
                The segmented vitiligo area in hand units.
                """
            )
            result_img = gr.Image(type="pil", label="Segmented vitiligo", height=400)
            vitiligo_area_in_hands = gr.Textbox(label="The vitiligo area in hand units")

    calibrate_btn.click(
        fn=calc_sticker_in_hand,
        inputs=hand_img,
        outputs=[segmented_hand_img, segmented_sticker_img, sticker_in_hands],
    )

    calculate_btn.click(
        fn=calc_vitiligo_area_in_hand,
        inputs=[area_img, sticker_in_hands, confidence, iou],
        outputs=[result_img, segmented_area_sticker_img, vitiligo_area_in_hands],
    )

if __name__ == "__main__":
    demo.launch(share=True)
