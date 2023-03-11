import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
# import mediapipe
import math
import os

# Image detection
model = YOLO('../Weights/yolov8m.pt')
# results = model('./Images/Shiva.jpeg', show=True)
# cv2.waitKey(0)

# Classnames
classnames = []
with open("../classnames.txt", 'rt') as f:
    classnames = f.readlines()
classnames = [cls.rstrip('\n') for cls in classnames]
print(classnames)


# Video detection
"""from WebCam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
"""

"""from Video"""
cap = cv2.VideoCapture("./Videos/traffic.mp4")

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)

            # x1, y1, w, h = box.xywh[0]
            w, h = x2 - x1, y2 - y1
            bbox = int(x1), int(y1), int(w), int(h)
            cvzone.cornerRect(img, bbox=bbox)

            # Confidence
            conf = math.ceil(box.conf[0] * 100) / 100
            print(f"Conf: {conf}")

            # Class
            cls_idx = int(box.cls[0])
            cls = classnames[cls_idx]

            # Display only cars bikes truck bus
            # if conf >= 0.3 and (cls == "car" or cls == "bus" or cls == "truck" or cls == "motorcycle"):
            #     cvzone.putTextRect(img, f'{cls} {conf}', (max(0, x1), max(35, y1)), scale=1)

            if conf >= 0.8 and (cls == "bus" or cls == "motorcycle"):
                cvzone.putTextRect(img, f'{cls} {conf}', (max(0, x1), max(35, y1)), scale=1)

    cv2.imshow("Video Frame", img)
    cv2.waitKey(1)








