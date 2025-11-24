from fastapi import FastAPI, UploadFile, File
from rfdetr import RFDETRNano
from PIL import Image
import io
import torch
import supervision as sv

app = FastAPI()

NUM_CLASSES = 1
WEIGHTS_PATH = "../weights/rf-detr-checkpoint_best_total.pth"

model = RFDETRNano(
	num_classes=NUM_CLASSES, 
	pretrain_weights=WEIGHTS_PATH
)
model.optimize_for_inference()

@app.get("/")
async def root():
	return {"message": "RFDETR Object Detection API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    detections = model.predict(image, threshold=0.5)

    results = []
    for xyxy, mask, confidence, class_id, tracker_id, data in detections:
        results.append({
            "class_id": int(class_id),
            "confidence": float(confidence),
            "bbox": xyxy.tolist()
        })

    return {"detections": results}