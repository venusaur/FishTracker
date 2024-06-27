import os
import cv2
from ultralytics import YOLO

# Video input and output paths
video_path = os.path.join('fish_white.m4v')
video_path_out = '{}_out.mp4'.format(video_path)
clip_output_dir = 'inactive_clips'

# Ensure the output directory exists
os.makedirs(clip_output_dir, exist_ok=True)

# Open the video capture
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape

# Initialize video writer for output
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# Path to the YOLO model weights
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')

# Load the YOLO model
model = YOLO(model_path)

# Detection threshold
threshold = 0.5

# Track inactivity state
inactivity_started = None
inactive_clip_count = 0

while ret:
    results = model(frame)[0]

    # Check for inactivity
    if all(score <= threshold for _, _, _, _, score, _ in results.boxes.data.tolist()):
        if inactivity_started is None:
            inactivity_started = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Start time of inactivity
    else:
        if inactivity_started is not None:
            # End of inactivity period
            inactivity_ended = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # End time of inactivity

            # Calculate duration of inactivity
            inactivity_duration = inactivity_ended - inactivity_started

            # If inactivity duration is greater than threshold, save the clip
            if inactivity_duration >= 5.0:  # Adjust as needed
                clip_output_path = os.path.join(clip_output_dir, f'inactive_clip_{inactive_clip_count}.mp4')
                inactive_clip_count += 1

                # Write the inactive clip to file
                inactive_clip_out = cv2.VideoWriter(clip_output_path, cv2.VideoWriter_fourcc(*'MP4V'),
                                                    int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
                start_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - int(cap.get(cv2.CAP_PROP_FPS) * 2)  # 2 second buffer
                start_frame = max(start_frame, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                while True:
                    ret, frame = cap.read()
                    if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > int(cap.get(cv2.CAP_PROP_FPS) * 10):
                        break
                    inactive_clip_out.write(frame)
                    out.write(frame)

                inactive_clip_out.release()

            inactivity_started = None

    # Draw bounding boxes and labels for visualization
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Write the frame to the output video
    out.write(frame)

    # Read the next frame
    ret, frame = cap.read()

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
