import os
from pathlib import Path
import torch
import supervision as sv
from PIL import Image
from rfdetr import RFDETRNano

from tongue_crack.UNet.unet import UNet
from .segment_image import segment_image

BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "SEGMENTATION_WEIGHTS": BASE_DIR / "weights/unet_best_model.pth",
    "DETECTION_WEIGHTS": BASE_DIR / "weights/rf-detr-checkpoint_best_total.pth",
    "IMAGE_PATH": BASE_DIR / "test-image/tongue.jpg",
    "OUTPUT_PATH": BASE_DIR / "test-image/output.jpg"
}

device = CONFIG["DEVICE"]

# load models
segmentation_model = UNet(in_channels=3, num_classes=1)
segmentation_model.load_state_dict(
    torch.load(str(CONFIG["SEGMENTATION_WEIGHTS"]), map_location=torch.device(device))
)

detection_model = RFDETRNano(
    num_classes=1, 
    pretrain_weights=str(CONFIG["DETECTION_WEIGHTS"])
)
detection_model.optimize_for_inference()

def detect_crack(image):
    print("Running inference...")

    segmented_image, xmin, ymin = segment_image(segmentation_model, image, device)
    detections = detection_model.predict(segmented_image, threshold=0.5)

    # Adjust detection coordinates to original image
    offsets = [xmin, ymin, xmin, ymin]
    detections.xyxy += offsets

    print("Inference completed.")

    return detections

def visualize_detections(image, detections, segmented_image):
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)

    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_scale=text_scale, smart_position=True)

    labels = [f"crack {confidence:0.2f}" for confidence in detections.confidence]

    annotated_image = image.copy()
    annotated_image = box_annotator.annotate(
        scene=annotated_image, 
        detections=detections,
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=detections,
        labels=labels
    )

    return annotated_image

if __name__ == "__main__":
    image_path = str(CONFIG["IMAGE_PATH"])
    output_path = str(CONFIG["OUTPUT_PATH"])

    image = Image.open(image_path)
    detections = detect_crack(image)
    result_image = visualize_detections(image, detections, None)
    result_image.save(output_path)
    print(f"Output saved to {output_path}")