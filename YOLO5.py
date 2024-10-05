import torch
import cv2
import numpy as np

# Load the YOLOv5 model (pre-trained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load an image
img_path = r'C:\Users\omash\OneDrive\Desktop\YOLO\Images\dogo.jpg.jpeg'

img = cv2.imread(img_path)

# Run inference
results = model(img)

# Get detected objects
detections = results.xyxy[0]  # x1, y1, x2, y2, confidence, class

# Draw bounding boxes on the image
for detection in detections:
    x1, y1, x2, y2, conf, cls = detection
    label = results.names[int(cls)]  # Get the label of the object
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    cv2.putText(img, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Show the image with bounding boxes
cv2.imshow('Object Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the output image
cv2.imwrite('output_image.jpg', img)
