# YOLOv8 Object Detection API

This project is a **web-based object detection tool** built using **YOLOv8**, **FastAPI**, and **OpenCV**. It allows users to upload images and get **detected objects with bounding boxes, class names, and confidence scores**. Annotated images can also be saved for visualization.

---

## **Features**
- Detect multiple objects in images using YOLOv8.
- Returns detection results in JSON format.
- Annotated images with bounding boxes are saved automatically.
- Easy-to-use API built with FastAPI.

---

## **Requirements**
- Python 3.8 or higher
- pip

Install required packages using:

```bash
pip install ultralytics fastapi uvicorn python-multipart
