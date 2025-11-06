import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os

from unet import UNet

def extract_image(image_pth, model_pth, output_pth, device):
    model = UNet(in_channels=1, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    img = Image.open(image_pth).convert("L")
    original_size = img.size # Store original size
    input_img = transform(img).float().to(device)
    input_img = input_img.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits = model(input_img)

        probabilities = torch.softmax(logits, dim=1)
        pred_mask = torch.argmax(probabilities, dim=1)

    pred_mask_np = pred_mask.squeeze(0).cpu().numpy()

    # Resize the mask to the original image size
    pred_mask_img = Image.fromarray((pred_mask_np * 255).astype(np.uint8))
    pred_mask_img = pred_mask_img.resize(original_size, Image.NEAREST)
    pred_mask_np_orig_size = np.array(pred_mask_img)

    # Apply the mask to the original image
    img_np = np.array(img)
    extracted_img_np = img_np * pred_mask_np_orig_size

    # Convert the numpy array back to PIL Image
    extracted_img = Image.fromarray(extracted_img_np.astype(np.uint8))

    # Find bounding box of the extracted region to crop tightly
    rows = np.any(extracted_img_np, axis=1)
    cols = np.any(extracted_img_np, axis=0)
    if np.any(rows) and np.any(cols):
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        extracted_img = extracted_img.crop((xmin, ymin, xmax, ymax))
    else:
        # If the mask is empty, save an empty image or handle as appropriate
        extracted_img = Image.new('L', original_size)


    # Save results
    os.makedirs(output_pth, exist_ok=True)
    extracted_save_path = os.path.join(output_pth, "extracted_image.png")

    extracted_img.save(extracted_save_path)

    print(f"Extracted image saved to: {extracted_save_path}")