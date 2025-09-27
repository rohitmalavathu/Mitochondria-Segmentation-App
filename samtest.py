import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

start_x, start_y = -1, -1
end_x, end_y = -1, -1
image = None
original_image = None
drawing = False

def click_event(event, x, y, flags, param):
    global start_x, start_y, end_x, end_y, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
        print(f"Started drawing at: ({start_x}, {start_y})")

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_image = image.copy()
            cv2.rectangle(temp_image, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow("Selected Image", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        print(f"Finished drawing at: ({end_x}, {end_y})")
        temp_image = image.copy()
        cv2.rectangle(temp_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.imshow("Selected Image", temp_image)
        confirm_bounding_box(temp_image)

def confirm_bounding_box(temp_image):
    print("Press 'Enter' to confirm or 'c' to cancel the bounding box.")
    while True:
        key = cv2.waitKey(0)
        if key == 13:
            print(f"Bounding box confirmed: Top-left: ({start_x}, {start_y}), Bottom-right: ({end_x}, {end_y})")
            break
        elif key == ord('c'):
            print("Bounding box selection canceled. You can try again.")
            cv2.imshow("Selected Image", image)
            break

def open_image():
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.tif")])

    if filepath:
        global image
        global original_image
        original_image = cv2.imread(filepath)
        image = cv2.resize(original_image, (512, 512))
        cv2.imshow("Selected Image", image)
        cv2.setMouseCallback("Selected Image", click_event)
        while True:
            key = cv2.waitKey(0)
            if key == 13:
                print("Enter key pressed.")
                cv2.destroyAllWindows()
                break

open_image()

x1, x2 = sorted([start_x, end_x])
y1, y2 = sorted([start_y, end_y])

modifiedx1 = x1 * 6
modifiedx2 = x2 * 6
modifiedy1 = y1 * 6
modifiedy2 = y2 * 6

cropped_image_og = original_image[modifiedy1:modifiedy2, modifiedx1:modifiedx2]
cropped_image = image[y1:y2, x1:x2]
original_height, original_width = cropped_image.shape[:2]
cropped_image = cv2.resize(cropped_image, (256, 256))

scale_x = original_width / 256
scale_y = original_height / 256
input_points = [[[128, 128]]]

FINE_TUNED_MODEL_WEIGHTS = hf_hub_download(repo_id="rohitmalavathu/SAM2FineTunedMito", filename="fine_tuned_sam2_2000.torch")
sam2_checkpoint = hf_hub_download(repo_id="rohitmalavathu/SAM2FineTunedMito", filename="sam2_hiera_small.pt")
model_cfg = "sam2_hiera_s.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cpu")

predictor = SAM2ImagePredictor(sam2_model)
predictor.model.load_state_dict(torch.load(FINE_TUNED_MODEL_WEIGHTS, map_location=torch.device('cpu')))

if cropped_image.ndim == 2:
    cropped_image = np.stack([cropped_image] * 3, axis=-1)

with torch.no_grad():
    predictor.set_image(cropped_image)
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=[[1]]
    )

sorted_masks = masks[np.argsort(scores)][::-1]
seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)

for i in range(sorted_masks.shape[0]):
    mask = sorted_masks[i]
    if (mask * occupancy_mask).sum() / mask.sum() > 0.15:
        continue

    mask_bool = mask.astype(bool)
    mask_bool[occupancy_mask] = False
    seg_map[mask_bool] = i + 1
    occupancy_mask[mask_bool] = True

seg_mask = gaussian_filter(seg_map.astype(float), sigma=2)
smoothed_mask = (seg_mask > 0.5).astype(np.uint8)
segmentation_resized = cv2.resize(smoothed_mask, (original_width, original_height))
image = image[:, :, 2]
segmentation_resized = segmentation_resized * 255

contours, _ = cv2.findContours(segmentation_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
segmentation_resized_rgb = cv2.cvtColor(segmentation_resized, cv2.COLOR_GRAY2BGR)
outline_image = cv2.drawContours(segmentation_resized_rgb, contours, -1, (255, 255, 0), 2)

yellow_lower = np.array([255, 255, 0])
yellow_upper = np.array([255, 255, 0])
mask_yellow = cv2.inRange(outline_image, yellow_lower, yellow_upper)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

image_copy = image.copy()
image_copy[y1:y2, x1:x2][mask_yellow == 255] = [0, 255, 255]
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

cropped_image_og_rgb = cv2.cvtColor(cropped_image_og, cv2.COLOR_BGR2RGB)
segmentation_for_overlay = cv2.resize(smoothed_mask, (cropped_image_og.shape[1], cropped_image_og.shape[0]))

red_mask = np.zeros_like(cropped_image_og_rgb)
red_mask[segmentation_for_overlay == 1] = [255, 0, 0]  

overlay = cropped_image_og_rgb.copy()
alpha = 0.25  
mask_bool = segmentation_for_overlay.astype(bool)
overlay[mask_bool] = cv2.addWeighted(cropped_image_og_rgb[mask_bool], 1-alpha, red_mask[mask_bool], alpha, 0)

plt.figure(figsize=(6, 6))
plt.subplot(1, 2, 1)
plt.title('Original with Yellow Outline')
plt.imshow(image_copy)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Cropped with Red Overlay')
plt.imshow(overlay)
plt.axis('off')

plt.tight_layout()
plt.show()