import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os

from unet import UNet

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

    # Resize the mask to the original image size
    pred_mask_img = Image.fromarray((pred_mask_np * 255).astype(np.uint8))
    pred_mask_img = pred_mask_img.resize(original_size, Image.NEAREST)
    pred_mask_np_orig_size = np.array(pred_mask_img) // 255

    # Apply the mask to the original image
    img_np = np.array(img)
    extracted_img_np = img_np * pred_mask_np_orig_size[:, :, np.newaxis]

    # Convert the numpy array back to PIL Image
    extracted_img = Image.fromarray(extracted_img_np.astype(np.uint8))

    # Find bounding box of the extracted region to crop
    margin = 5

    rows = np.any(extracted_img_np, axis=1)
    cols = np.any(extracted_img_np, axis=0)
    if np.any(rows) and np.any(cols):
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        xmin = max(0, xmin - margin)
        ymin = max(0, ymin - margin)
        xmax = min(original_size[0], xmax + margin)
        ymax = min(original_size[1], ymax + margin)

        extracted_img = extracted_img.crop((xmin, ymin, xmax, ymax))
    else:
        print("No object founded in the image.")
        extracted_img = img

    # Save results
    os.makedirs(output_pth, exist_ok=True)
    extracted_save_path = os.path.join(output_pth, img_name)

    extracted_img.save(extracted_save_path)

    print(f"Extracted image saved to: {extracted_save_path}")

if __name__ == "__main__":
    # IMAGE_PATH = "./test/tongue2.jpg"
    IMAGE_PATH = "./test_images"
    MODEL_PATH = "./checkpoints/unet_best_model.pth"
    OUTPUT_PATH = "./outputs"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for img_name in os.listdir(IMAGE_PATH):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(IMAGE_PATH, img_name)
            print(f"Processing image: {image_path}")

            extract_image(image_path, img_name, MODEL_PATH, OUTPUT_PATH, device)