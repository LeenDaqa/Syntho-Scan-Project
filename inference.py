import os
import numpy as np
import torch
from medpy.filter.binary import largest_connected_component
from skimage.io import imsave
from PIL import Image

from seg_dataset import BrainSegmentationDataset as Dataset
from seg_unet import UNet

def load_image(image_path):
    try:
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = image.resize((256, 256))
        image_array = np.array(image)
        image_array = np.transpose(image_array, (2, 0, 1))
        return image_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def segment_image(image_path, predictions_dir, weights_path, device="cuda:0"):
    os.makedirs(predictions_dir, exist_ok=True)
    device = torch.device("cpu" if not torch.cuda.is_available() else device)

    image = load_image(image_path)
    if image is None:  # Ensure the image loaded successfully
        print("Image loading failed.")
        return
    image_tensor = torch.tensor(image, dtype=torch.float).unsqueeze(0).to(device)

    with torch.no_grad():
        unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
        state_dict = torch.load(weights_path, map_location=device)
        unet.load_state_dict(state_dict)
        unet.eval()
        unet.to(device)

        y_pred = unet(image_tensor)
        y_pred_np = y_pred.cpu().squeeze().numpy()

        if y_pred_np.ndim != 2:
            raise ValueError("Prediction must be a 2-dimensional array.")

        thresholded_pred = y_pred_np > 0.9  # Thresholding
        segmented_image = largest_connected_component(thresholded_pred)

        segmented_image = segmented_image * 255
        segmented_image = segmented_image.astype(np.uint8)
        
         # Initialize the counter if not already done

        # Adjusted format to match the given structure
        filepath = os.path.join(predictions_dir, os.path.basename(image_path).replace('.png', '_segmented.png' ))
        imsave(filepath, segmented_image)
        print(f"Segmentation saved to {filepath}")

        return segment_image



