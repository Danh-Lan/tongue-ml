import torch
import numpy as np
from tqdm import tqdm

from data_loader import Dataset
from unet import UNet

def IoU(model, dataset):
    model.eval()

    intersection = 0
    union = 0

    with torch.no_grad():
        for img, true_mask in tqdm(dataset, desc="Calculating IoU"):
            img = img.to(device).unsqueeze(0)
            true_mask = true_mask.to(device)

            logits = model(img)
            probabilities = torch.sigmoid(logits)
            pred_mask = (probabilities > 0.5).float().squeeze()

            pred_inds = (pred_mask == 1)
            target_inds = (true_mask == 1)

            intersection += (pred_inds & target_inds).sum().item()
            union += (pred_inds | target_inds).sum().item()

    if union == 0:
        iou = 0.0
    else :
        iou = intersection / union

    return iou

def dice(model, dataset):
    model.eval()

    dice_score = 0
    num_samples = len(dataset)

    with torch.no_grad():
        for img, true_mask in tqdm(dataset, desc="Calculating Dice Score"):
            img = img.to(device).unsqueeze(0)
            true_mask = true_mask.to(device)

            logits = model(img)
            probabilities = torch.sigmoid(logits)
            pred_mask = (probabilities > 0.5).float().squeeze()

            intersection = (pred_mask * true_mask).sum().item()
            dice_score += (2.0 * intersection) / (pred_mask.sum().item() + true_mask.sum().item() + 1e-6)

    dice_score /= num_samples
    return dice_score

if __name__ == "__main__":
    TEST_PATH = "./data/test"
    dataset = Dataset(TEST_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(
        torch.load("./checkpoints/unet_best_model.pth", map_location=torch.device(device))['model_state_dict']
    )

    iou = IoU(model, dataset)
    dice_score = dice(model, dataset)

    print(f"  IoU: {iou:.4f}")
    print(f"  Dice Score: {dice_score:.4f}")