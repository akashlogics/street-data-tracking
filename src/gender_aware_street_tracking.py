import cv2
import torch
import numpy as np
from ultralytics import YOLO
from sort import *
from openpyxl import Workbook
from datetime import datetime

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize SORT tracker
mot_tracker = Sort()

# Initialize Excel workbook
wb = Workbook()
ws = wb.active
ws.append(["Timestamp", "Person ID", "Gender", "From Street", "To Street"])

# Define street areas (example coordinates, adjust as needed)
streets = {
    "Street1": [(0, 0), (320, 720)],
    "Street2": [(320, 0), (640, 720)],
    "Street3": [(640, 0), (960, 720)]
}

def get_street(x, y):
    for street, coords in streets.items():
        if coords[0][0] <= x <= coords[1][0] and coords[0][1] <= y <= coords[1][1]:
            return street
    return "Unknown"

# Initialize person tracking
person_tracks = {}

# Open video capture
cap = cv2.VideoCapture('path_to_your_video.mp4')  # Replace with your video file or camera index

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)
    detections = results[0].boxes.data

    # Filter for person class (class 0 in COCO dataset)
    person_detections = detections[detections[:, 5] == 0]

    # Update SORT tracker
    tracked_objects = mot_tracker.update(person_detections[:, :5].cpu())

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        current_street = get_street(cx, cy)
        
        # Determine gender
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        gender_results = model(crop)
        gender_classes = gender_results[0].boxes.cls
        gender = "Woman" if 0 in gender_classes else "Man"
        
        if obj_id not in person_tracks:
            person_tracks[obj_id] = {"last_street": current_street, "movements": [], "gender": gender}
        elif current_street != person_tracks[obj_id]["last_street"]:
            movement = (person_tracks[obj_id]["last_street"], current_street)
            person_tracks[obj_id]["movements"].append(movement)
            
            # Record movement in Excel
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ws.append([timestamp, int(obj_id), person_tracks[obj_id]["gender"], movement[0], movement[1]])
            wb.save("gender_aware_street_movements.xlsx")
            
            person_tracks[obj_id]["last_street"] = current_street

        # Draw bounding box, ID, and gender
        color = (0, 255, 0) if gender == "Woman" else (0, 0, 255)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"ID: {int(obj_id)} ({gender})", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow('Gender-Aware Street Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()