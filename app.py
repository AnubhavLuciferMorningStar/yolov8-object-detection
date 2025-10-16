from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import io
from PIL import Image

app = FastAPI(title="YOLOv8 Object Detection")

# Load the pretrained model (choose class: person, car, bottle etc.)
model = YOLO("yolov8n.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Read the image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Run YOLOv8 inference
    results = model.predict(image, conf=0.4)

    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]
        conf = round(float(box.conf), 2)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({
            "class_name": cls_name,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })

    # Save image with bounding boxes (optional for demo)
    annotated_img = results[0].plot()
    cv2.imwrite(f"output_images/detected_{file.filename}", annotated_img)

    return JSONResponse(content={"detections": detections})
