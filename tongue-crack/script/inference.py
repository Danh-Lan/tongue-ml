import torch
import supervision as sv
from PIL import Image
from rfdetr import RFDETRNano

NUM_CLASSES = 1
weights_path = "../weights/rf-detr-checkpoint_best_total.pth"
image_path = "../test-image/tongue.jpg"

model = RFDETRNano(
    num_classes=NUM_CLASSES,
    pretrain_weights=weights_path
)
model.optimize_for_inference()

# 3. INFERENCE
print("Running inference...")
image = Image.open(image_path)

# The rfdetr library returns sv.Detections object directly
detections = model.predict(image, threshold=0.5)

# 4. ANNOTATE & SAVE
annotator = sv.BoxAnnotator()
annotated_image = annotator.annotate(scene=image.copy(), detections=detections)

output_path = "../test-image/output.jpg"
annotated_image.save(output_path)
print(f"Saved result to {output_path}")