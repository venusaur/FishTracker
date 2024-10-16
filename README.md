
# Need to upload updating camera code....

# Fish Tracking and Inactivity Detection

## Overview

This script uses the YOLO object detection model to track fish in a video and log instances of inactivity. The code detects fish, tracks their positions, and identifies if any fish remain inactive for a specified duration. It then logs these instances to a file and highlights inactive fish in the video output.

## To Do
- Train a better model for higher detection rates
- Fix current inactivity tracker and ensure correct outputs
- Implement a GUI to control various functions (Recording, time ranges, etc)


## Prerequisites

Before running the script, ensure you have the following dependencies installed:

- Python 3.x
- OpenCV (`cv2`)
- Ultralytics YOLO library

You can install the necessary libraries using `pip`:

```bash
pip install opencv-python ultralytics
```
## Setup

  1. Download or train a YOLO model:

     Place your trained model weights (last.pt) in the runs/detect/train/weights/ directory. Ensure the model is trained to detect fish.
    
  3. Organize your video:

     Place the video file (fish.m4v) in the videos directory. The script will process this video.

## Configuration

Modify the following parameters as needed:

  - VIDEOS_DIR: Directory containing the input video.
  - video_path: Path to the input video file.
  - model_path: Path to the YOLO model weights.

```python
VIDEOS_DIR = os.path.join('.', 'videos')
video_path = os.path.join('fish.m4v')
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
```
  Detection Threshold: The confidence threshold for detecting fish. Default is 0.5.
  
  Inactive Time Threshold: The duration (in seconds) to consider a fish as inactive. Default is 60 seconds.

## How It Works

  Initialization: Load the video and model. Set up the video writer for output.
  Detection Loop:
      
  - Read frames from the video.
    - Use the YOLO model to detect fish in each frame.
    - Track the position of detected fish and check their activity status.
    - Log inactivity if a fish remains stationary for the defined threshold.
    - Draw bounding boxes and labels on the video frames.
    
  Output: Save the processed video with detected fish and their activity status.

## Execution

Run the script using Python:

```bash
python3 zebrafish_tracking.py
```

This will generate an output video ({filename}_out.mp4) and an inactivity log ({filename}_inactivity_log.txt) in the same directory.
Example Output

  - Output Video: Shows fish with bounding boxes and labels indicating inactivity.
  
  - Inactivity Log: Contains timestamps and positions of inactive fish.

### Notes

  1. Ensure the input video is in the correct directory.
  2. Adjust the detection threshold and inactivity threshold according to your needs.
  3. This script assumes fish are the only objects of interest. Modify the class_id and results.names as per your model’s class labels.
