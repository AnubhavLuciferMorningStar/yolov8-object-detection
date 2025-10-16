from ultralytics import YOLO
import cv2
import glob

model = YOLO("yolov8n.pt")

for img_path in glob.glob("test_images/*.jpg"):
    results = model.predict(img_path, conf=0.4)
    annotated_img = results[0].plot()
    out_path = "output_images/" + img_path.split("/")[-1]
    cv2.imwrite(out_path, annotated_img)
    print(f"Saved: {out_path}")
