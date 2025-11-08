import torch
import numpy as np
from tqdm import tqdm

from data_loader import Dataset
from unet import UNet

def IoU(model, dataset, num_classes):
    model.eval()

    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)

    with torch.no_grad():
        for img, true_mask in tqdm(dataset, desc="Calculating IoU"):
            img = img.to(device).unsqueeze(0)
            true_mask = true_mask.to(device)

            logits = model(img)
            probabilities = torch.softmax(logits, dim=1)
            pred_mask = torch.argmax(probabilities, dim=1).squeeze(0)

            for i in range(1, num_classes): # Skip background class
                pred_inds = (pred_mask == i)
                target_inds = (true_mask == i)

                intersection[i] += (pred_inds & target_inds).sum().item()
                union[i] += (pred_inds | target_inds).sum().item()
    
    # Calculate IoU for each class
    iou = np.divide(intersection, union, where=union != 0)
    
    return iou

if __name__ == "__main__":
    TEST_PATH = "./data/test"
    dataset = Dataset(TEST_PATH)

    NUM_CLASSES = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load("./checkpoints/unet_best_model.pth", map_location=device))

    iou_per_class = IoU(model, dataset, NUM_CLASSES)

    for i, iou in enumerate(iou_per_class):
        if i == 0:
            continue  # Skip background class
        
        print(f"  IoU for class {i}: {iou:.4f}")