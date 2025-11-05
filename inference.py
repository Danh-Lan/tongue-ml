import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os

from unet import UNet

def single_image_inference(image_pth, model_pth, output_pth, device):
    model = UNet(in_channels=1, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    img = Image.open(image_pth).convert("L")
    input_img = transform(img).float().to(device)
    input_img = input_img.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits = model(input_img)
        
        probabilities = torch.softmax(logits, dim=1)
        pred_mask = torch.argmax(probabilities, dim=1)

    pred_mask_np = pred_mask.squeeze(0).cpu().numpy()    
    mask_pil = Image.fromarray((pred_mask_np * 255).astype(np.uint8))

    # Save results
    os.makedirs(output_pth, exist_ok=True)
    input_save_path = os.path.join(output_pth, "input_image.png")
    mask_save_path = os.path.join(output_pth, "predicted_mask.png")

    img.save(input_save_path)
    mask_pil.save(mask_save_path)

    print(f"Input image saved to: {input_save_path}")
    print(f"Binary mask saved to: {mask_save_path}")

if __name__ == "__main__":
    SINGLE_IMG_PATH = "./test/tongue.jpg"
    MODEL_PATH = "./checkpoints/unet_epoch_10.pth"
    OUTPUT_PATH = "./test"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    single_image_inference(SINGLE_IMG_PATH, MODEL_PATH, OUTPUT_PATH, device)