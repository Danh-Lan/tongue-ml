import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from scipy.ndimage import label

from unet import UNet

def crop_image(img, mask, margin=5):
    img_np = np.array(img)
    mask_np = np.array(mask)

    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)

    # margin to avoid cutting too close to the object
    if np.any(rows) and np.any(cols):
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        xmin = max(0, xmin - margin)
        ymin = max(0, ymin - margin)
        xmax = min(img_np.shape[1], xmax + margin)
        ymax = min(img_np.shape[0], ymax + margin)

        cropped_img = img.crop((xmin, ymin, xmax, ymax))
        return cropped_img
    else:
        print("No object founded in the image.")
        return img

def extract_image(model, image_pth, img_name, output_pth, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    img = Image.open(image_pth).convert("RGB")
    original_size = img.size
    input_img = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_img)
        probabilities = torch.sigmoid(logits)
        pred_mask = (probabilities > 0.5).float()

    pred_mask_np = pred_mask.squeeze().cpu().numpy()

    # Find the largest connected component in the mask
    labeled_mask, num_features = label(pred_mask_np)
    if num_features > 0:
        component_areas = [(labeled_mask == i).sum() for i in range(1, num_features + 1)]
        largest_component = np.argmax(component_areas) + 1  # Component labels start at 1
        pred_mask_np = (labeled_mask == largest_component).astype(np.uint8)

    # Resize the mask to the original image size
    pred_mask_img = Image.fromarray((pred_mask_np * 255).astype(np.uint8))
    pred_mask_img = pred_mask_img.resize(original_size, Image.NEAREST)
    pred_mask_np_orig_size = np.array(pred_mask_img) // 255

    # Apply the mask to the original image
    img_np = np.array(img)
    extracted_img_np = img_np * pred_mask_np_orig_size[:, :, np.newaxis]

    # Convert the numpy array back to PIL Image
    extracted_img = Image.fromarray(extracted_img_np.astype(np.uint8))

    # Crop the image to the bounding box of the extracted area
    extracted_img = crop_image(extracted_img, pred_mask_img)

    # Save results
    os.makedirs(output_pth, exist_ok=True)
    extracted_save_path = os.path.join(output_pth, img_name)

    extracted_img.save(extracted_save_path)

    print(f"Extracted image saved to: {extracted_save_path}")