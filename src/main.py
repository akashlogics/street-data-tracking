import cv2
import numpy as np
from ultralytics import YOLO
from openpyxl import Workbook
from datetime import datetime, timedelta
from collections import deque

def draw_street_boundaries(frame):
    for street, (start, end) in streets.items():
        cv2.line(frame, (start[0], 0), (start[0], 720), (255, 255, 255), 2)
        cv2.putText(frame, f"Street {street}", (start[0] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def classify_person(height, width):
    aspect_ratio = height / width
    if aspect_ratio > 2.2:  
        return "Man"
    else:
        return "Woman"

# Initialize YOLO model
model = YOLO('yolov8n.pt')

# Define street areas (you may need to adjust these based on your video)
streets = {
    1: [(0, 0), (320, 720)],
    2: [(320, 0), (640, 720)],
    3: [(640, 0), (960, 720)],
    4: [(960, 0), (1280, 720)]
}

# Initialize Excel workbook
wb = Workbook()
ws = wb.active
ws.append(["Timestamp", "Gender", "Mode", "From Street", "To Street"])

# Initialize video capture
cap = cv2.VideoCapture('street_camera.mp4')
start_time = datetime.now()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter('output_gender_tracking.mp4', fourcc, fps, (width, height))

# Object tracking
object_tracker = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    draw_street_boundaries(frame)

    # Perform YOLO detection
    results = model(frame)

    current_time = start_time + timedelta(milliseconds=cap.get(cv2.CAP_PROP_POS_MSEC))
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    # Detect bicycles first
    bicycles = []
    for det in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 1:  # 1 is the class for bicycle in COCO dataset
            bicycles.append((int(x1), int(y1), int(x2), int(y2)))

    for det in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Only track people (class 0 in COCO dataset)
        if int(cls) != 0:
            continue

        # Determine gender based on aspect ratio
        height = y2 - y1
        width = x2 - x1
        category = classify_person(height, width)

        # Determine if the person is cycling
        is_cycling = any(
            abs(x1 - bx1) < 50 and abs(y1 - by1) < 50
            for bx1, by1, bx2, by2 in bicycles
        )
        mode = "Cycling" if is_cycling else "Walking"

        # Determine current street
        center_x = (x1 + x2) // 2
        current_street = next(street for street, (start, end) in streets.items() if start[0] <= center_x < end[0])

        object_id = f"{x1}_{y1}"
        if object_id not in object_tracker:
            object_tracker[object_id] = {
                'street': current_street,
                'last_seen': current_time,
                'color': (0, 255, 0) if category == "Man" else (0, 0, 255),  # Green for men, Red for women
                'trail': deque(maxlen=20),
                'category': category,
                'mode': mode
            }
        
        tracker = object_tracker[object_id]
        tracker['trail'].appendleft((center_x, (y1 + y2) // 2))
        tracker['mode'] = mode  # Update mode in case it changed
        
        # Record all movements
        ws.append([timestamp, tracker['category'], tracker['mode'], tracker['street'], current_street])
        
        if current_street != tracker['street']:
            tracker['street'] = current_street

        # Draw bounding box and trail
        cv2.rectangle(frame, (x1, y1), (x2, y2), tracker['color'], 2)
        for i in range(1, len(tracker['trail'])):
            if tracker['trail'][i - 1] is None or tracker['trail'][i] is None:
                continue
            cv2.line(frame, tracker['trail'][i - 1], tracker['trail'][i], tracker['color'], 2)
        
        # Display object info
        label = f"{tracker['category']} {tracker['mode']}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tracker['color'], 2)

    # Add legend
    cv2.putText(frame, "Legend:", (10, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Man (Walking/Cycling)", (10, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "Woman (Walking/Cycling)", (10, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Display the frame
    cv2.imshow('People Tracking', cv2.resize(frame, (1280, 720)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

out = cv2.VideoWriter('output_gender_tracking.mp4', fourcc, fps, (width, height))
# Save Excel file
wb.save('people_tracking_data.xlsx')

