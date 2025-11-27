import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.ndimage import label

def crop_image(img, mask):
    img_np = np.array(img)
    mask_np = np.array(mask)

    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)

    if np.any(rows) and np.any(cols):
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        cropped_img = img.crop((xmin, ymin, xmax, ymax))
        # xmin, ymin for detection box adjustment
        return cropped_img, xmin, ymin
    else:
        print("No object founded in the image.")
        return img, -1, -1

def segment_image(model, img, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

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
    extracted_img, xmin, ymin = crop_image(extracted_img, pred_mask_img)

    return extracted_img, xmin, ymin