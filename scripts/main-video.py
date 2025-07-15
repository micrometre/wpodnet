import os
import sys
import cv2
from local_utils import load_model, detect_lp, im2single
import numpy as np
import uuid

# Create directories if they don't exist
if not os.path.exists("images"):
    os.makedirs("images")
if not os.path.exists("images/warped"):
    os.makedirs("images/warped")        
if not os.path.exists("images/original"):
    os.makedirs("images/original")

# Path to the input video
if len(sys.argv) > 1:
    video_path = sys.argv[1]
else:
    print("Error: Please provide a video file path as argument")
    sys.exit(1)

# Generate unique ID for this detection session
session_id = str(uuid.uuid4())

# Load the pretrained WPOD-NET model
model_path = 'wpod-net.json'    # Model architecture file
weights_path = 'wpod-net.h5'    # Model weights file
wpod_net = load_model(model_path, weights_path)

# Video capture parameters
Dmax = 608
Dmin = 288

# Open video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file")
    sys.exit(1)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Processing video: {video_path}")
print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {frame_count}")

# Process each frame
frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_number += 1
    print(f"\nProcessing frame {frame_number}/{frame_count}")
    
    # Generate unique ID for this frame's detection
    detection_id = f"{session_id}_frame{frame_number}"
    
    # Display original frame
    cv2.imwrite(f"images/original/{detection_id}.jpg", frame)
    
    # Calculate resize ratio and bounding dimension
    ratio = float(max(frame.shape[:2])) / min(frame.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    
    try:
        # Detect license plate(s) in the frame
        _, LpImg, lp_type = detect_lp(
            wpod_net, im2single(frame), bound_dim, lp_threshold=0.5
        )
        
        # If any license plate images are detected
        if len(LpImg):
            print(f"Detected {len(LpImg)} license plate(s) in frame {frame_number}")
            for i, lp_img in enumerate(LpImg):
                lp_img_uint8 = (lp_img * 255).astype(np.uint8)
                lp_bgr = cv2.cvtColor(lp_img_uint8, cv2.COLOR_RGB2BGR)
                
                # Display and save each detected license plate
                cv2.imwrite(f"images/warped/{detection_id}_lp{i}.jpg", lp_bgr)
                
    except Exception as e:
        print(f"Error processing frame {frame_number}: {str(e)}")
    
    # Check for quit command (press 'q' to exit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("\nVideo processing completed")