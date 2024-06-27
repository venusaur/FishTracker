import os
import time
from ultralytics import YOLO
import cv2

VIDEOS_DIR = os.path.join('.', 'videos')

video_path = os.path.join('fish_white.m4v')
video_path_out = '{}_out.mp4'.format(video_path)
log_path = '{}_inactivity_log.txt'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

if not ret:
    raise ValueError("Failed to open video file. Check if the file exists and is accessible.")

H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5
fps = cap.get(cv2.CAP_PROP_FPS)
inactive_time_threshold = 60  # in seconds
inactive_frame_count_threshold = int(fps * inactive_time_threshold)

fish_positions = {}
inactive_fish = set()

movement_threshold = 20  # Adjust as per your scenario

def get_center(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def log_inactivity(fish_id, center, frame_number):
    timestamp = frame_number / fps
    with open(log_path, 'a') as log_file:
        log_file.write(f"Time: {timestamp:.2f}s, Fish ID: {fish_id}, State: INACTIVE, Position: {center}\n")

frame_number = 0

# Adjusted inactive time thresholds
inactive_time_threshold_min = 60  # 1 minute
inactive_time_threshold_max = 300  # 5 minutes

# Calculate frame count thresholds
inactive_frame_count_threshold_min = int(fps * inactive_time_threshold_min)
inactive_frame_count_threshold_max = int(fps * inactive_time_threshold_max)

# Inside the while loop where you track fish movement
while ret:
    results = model(frame)[0]
    current_fish_positions = []

    if frame is None:
        break  # Exit the loop if there are no more frames to read

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            fish_id = int(class_id)
            center = get_center(x1, y1, x2, y2)
            current_fish_positions.append((fish_id, center))

            if fish_id not in fish_positions:
                fish_positions[fish_id] = []

            fish_positions[fish_id].append(center)

            # Track movement
            if len(fish_positions[fish_id]) > inactive_frame_count_threshold_max:
                fish_positions[fish_id].pop(0)
                
                if len(fish_positions[fish_id]) > 1:
                    last_position = fish_positions[fish_id][-2]
                    current_position = fish_positions[fish_id][-1]
                    distance = ((current_position[0] - last_position[0])**2 + (current_position[1] - last_position[1])**2) ** 0.5
                    
                    if distance < movement_threshold and fish_id not in inactive_fish:
                        inactive_fish.add(fish_id)
                        log_inactivity(fish_id, current_position, frame_number)
                    elif distance >= movement_threshold and fish_id in inactive_fish:
                        inactive_fish.remove(fish_id)

                    # Check for long-term inactivity (between 1 to 5 minutes)
                    if len(fish_positions[fish_id]) > inactive_frame_count_threshold_min and len(fish_positions[fish_id]) <= inactive_frame_count_threshold_max:
                        if fish_id not in inactive_fish:
                            inactive_fish.add(fish_id)
                            log_inactivity(fish_id, current_position, frame_number)
                        elif distance >= movement_threshold and fish_id in inactive_fish:
                            inactive_fish.remove(fish_id)

            # Draw rectangle and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            label = results.names[int(class_id)].upper()
            if fish_id in inactive_fish:
                label += " - INACTIVE"
            cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()
    frame_number += 1

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
