import os
import sys
import cv2
from local_utils import load_model, detect_lp, im2single
import numpy as np
import uuid

if not os.path.exists("images"):
    os.makedirs("images")
if not os.path.exists("images/warped"):
    os.makedirs("images/warped")        
if not os.path.exists("images/original"):
    os.makedirs("images/original")

# Path to the input image
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    img_path = "test/2.jpg"  # Default image if no argument is provided


# Generate unique ID for this detection
detection_id = str(uuid.uuid4())



# Load the pretrained WPOD-NET model
model_path = 'wpod-net.json'    # Model architecture file
weights_path = 'wpod-net.h5'    # Model weights file
wpod_net = load_model(model_path, weights_path)

# Read the input vehicle image
Ivehicle = cv2.imread(img_path)
cv2.imshow("Original image", Ivehicle)  # Display the original image
cv2.imwrite(f"images/original/{detection_id}.jpg", Ivehicle)  # Save the detected license plate


# Set maximum and minimum dimensions for license plate detection
Dmax = 608
Dmin = 288

# Calculate resize ratio and bounding dimension
ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
side = int(ratio * Dmin)
bound_dim = min(side, Dmax)

# Detect license plate(s) in the image
_, LpImg, lp_type = detect_lp(
    wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5
)

# If any license plate images are detected, display the first one
if len(LpImg):
    print("Detected license plate image shape:", LpImg[0].shape)
    print("Min pixel value:", LpImg[0].min(), "Max pixel value:", LpImg[0].max())
    lp_img_uint8 = (LpImg[0] * 255).astype(np.uint8)
    lp_bgr = cv2.cvtColor(lp_img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imshow(f"detected Licence plate - {detection_id}", lp_bgr)
    cv2.imwrite(f"images/warped/{detection_id}.jpg", lp_bgr)  # Save the detected license plate
    cv2.waitKey(0)

# Close all OpenCV windows
if cv2.getWindowProperty(f"detected Licence plate - {detection_id}", cv2.WND_PROP_VISIBLE) >= 1:
    cv2.destroyAllWindows()