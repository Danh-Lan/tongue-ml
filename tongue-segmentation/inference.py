import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from scipy.ndimage import label

from unet import UNet

def image_to_tensor(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    output = transform(image)
    return output

def predict(model, image_pth, output_pth, device):
    input_img = image_to_tensor(image_pth).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_img)
        probabilities = torch.sigmoid(logits)
        # np.savetxt("probabilities.txt", probabilities.squeeze().cpu().numpy())
        pred_mask = (probabilities > 0.5).float()

    pred_mask_np = pred_mask.squeeze().cpu().numpy()

    labeled_mask, num_features = label(pred_mask_np)
    if num_features > 0:
        component_areas = [(labeled_mask == i).sum() for i in range(1, num_features + 1)]
        largest_component = np.argmax(component_areas) + 1  # Component labels start at 1
        pred_mask_np = (labeled_mask == largest_component).astype(np.uint8)

    mask_pil = Image.fromarray((pred_mask_np * 255).astype(np.uint8))

    # Save results
    os.makedirs(output_pth, exist_ok=True)
    input_save_path = os.path.join(output_pth, "input_image.jpg")
    mask_save_path = os.path.join(output_pth, "predicted_mask.jpg")

    mask_pil.save(mask_save_path)
    print(f"Binary mask saved to: {mask_save_path}")