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

def extract_image(image_pth, img_name, model_pth, output_pth, device):
    model = UNet(in_channels=3, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    img = Image.open(image_pth).convert("RGB")
    original_size = img.size
    input_img = transform(img).float().unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_img)
        probabilities = torch.softmax(logits, dim=1)
        pred_mask = torch.argmax(probabilities, dim=1)

    pred_mask_np = pred_mask.squeeze(0).cpu().numpy()

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

if __name__ == "__main__":
    IMAGE_PATH = "./test_images"
    MODEL_PATH = "./checkpoints/unet_best_model.pth"
    OUTPUT_PATH = "./outputs"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for img_name in os.listdir(IMAGE_PATH):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(IMAGE_PATH, img_name)
            print(f"Processing image: {image_path}")
        
            extract_image(image_path, img_name, MODEL_PATH, OUTPUT_PATH, device)