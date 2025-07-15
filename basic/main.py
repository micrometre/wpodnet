import cv2
from local_utils import load_model, detect_lp, im2single
import numpy as np

# Path to the input image
img_path = "test/2.jpg"

# Load the pretrained WPOD-NET model
model_path = 'wpod-net.json'    # Model architecture file
weights_path = 'wpod-net.h5'    # Model weights file
wpod_net = load_model(model_path, weights_path)

# Read the input vehicle image
Ivehicle = cv2.imread(img_path)
cv2.imshow("Original image", Ivehicle)  # Display the original image
cv2.imwrite("original_image.jpg", Ivehicle)  # Save the original image
cv2.waitKey(0)

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
    cv2.imshow("detected Licence plate", lp_bgr)
    cv2.imwrite("detected_license_plate.jpg", lp_bgr)  # Save the detected license plate
    cv2.waitKey()

# Close all OpenCV windows
cv2.destroyAllWindows()