import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os

from unet import UNet

def image_to_tensor(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    output = transform(image).float()
    return output

def predict(image_pth, model_pth, output_pth, device):
    model = UNet(in_channels=3, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    input_img = image_to_tensor(image_pth).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_img)
        probabilities = torch.softmax(logits, dim=1)
        pred_mask = torch.argmax(probabilities, dim=1)

    pred_mask_np = pred_mask.squeeze(0).cpu().numpy()
    mask_pil = Image.fromarray((pred_mask_np * 255).astype(np.uint8))

    # Save results
    os.makedirs(output_pth, exist_ok=True)
    input_save_path = os.path.join(output_pth, "input_image.jpg")
    mask_save_path = os.path.join(output_pth, "predicted_mask.jpg")

    mask_pil.save(mask_save_path)
    print(f"Binary mask saved to: {mask_save_path}")

if __name__ == "__main__":
    # SINGLE_IMG_PATH = "./test/tongue.jpg"
    SINGLE_IMG_PATH = "./data/test/-0011_51_-__jpg.rf.60a43eeb0ea548e7f1cd4431a2f5ee71.jpg"
    MODEL_PATH = "./checkpoints/unet_best_model.pth"
    OUTPUT_PATH = "./test"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predict(SINGLE_IMG_PATH, MODEL_PATH, OUTPUT_PATH, device)