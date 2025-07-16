import os
import sys
import cv2
from local_utils import load_model, detect_lp, im2single
import numpy as np
import uuid
import easyocr

if not os.path.exists("images"):
    os.makedirs("images")
if not os.path.exists("images/warped"):
    os.makedirs("images/warped")        
if not os.path.exists("images/original"):
    os.makedirs("images/original")

os.makedirs('outputs', exist_ok=True)

# Path to the input file (image or video)
if len(sys.argv) > 1:
    input_path = sys.argv[1]
else:
    input_path = "test/2.jpg"  # Default image if no argument is provided

# Optional: process every nth frame (default 1)
nth = 25
if len(sys.argv) > 2:
    try:
        nth = int(sys.argv[2])
        if nth < 1:
            nth = 1
    except ValueError:
        nth = 1

# Load the pretrained WPOD-NET model
model_path = 'wpod-net.json'    # Model architecture file
weights_path = 'wpod-net.h5'    # Model weights file
wpod_net = load_model(model_path, weights_path)

# Set maximum and minimum dimensions for license plate detection
Dmax = 608
Dmin = 288



detection_id = str(uuid.uuid4())

reader = easyocr.Reader(['en'], gpu=False)

def process_frame(Ivehicle, detection_id_prefix=""):
    if Ivehicle is None or Ivehicle.size == 0 or min(Ivehicle.shape[:2]) < 30:
        return False
    # Save original frame
    if detection_id_prefix:
        orig_path = f"images/original/{detection_id_prefix}.jpg"
    else:
        orig_path = f"images/original/{str(uuid.uuid4())}.jpg"
    cv2.imwrite(orig_path, Ivehicle)
    # Calculate resize ratio and bounding dimension
    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    # Detect license plate(s) in the image
    try:
        _, LpImg, lp_type = detect_lp(
            wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5
        )
    except IndexError:
        # No license plate candidates found
        return False
    # If any license plate images are detected, save the first one
    if len(LpImg):
        lp_img_uint8 = (LpImg[0] * 255).astype(np.uint8)
        lp_bgr = cv2.cvtColor(lp_img_uint8, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(lp_bgr, cv2.COLOR_RGB2GRAY)
        # Contrast enhancement with CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray) 
        
        
        # 2. Otsu's thresholding
        _, binary_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Choose the best one based on your typical plates
        binary = binary_otsu  # or binary_otsu
        
        # Morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Denoising
        denoised = cv2.fastNlMeansDenoising(binary, h=10)

        if detection_id_prefix:
            warped_path = f"images/warped/{detection_id_prefix}.jpg"
        cv2.imwrite(warped_path, denoised)
        # OCR with easyocr
        result = reader.readtext(denoised)
        if result:
            plate_texts = [r[1] for r in result]


            plate_info = {
            "uuid": detection_id,
            "results": [
                {
                    "plate": plate_texts,
                    "image_path": str(orig_path),
                    "original_frame_path": orig_path
                }
            ]
            }
            print(plate_info)



            print(f"[OCR] License plate(s) detected: {plate_texts}")
        else:
            print("[OCR] No text detected on license plate.")
        return True
    return False

# Check if input is a video file
if input_path.lower().endswith('.mp4'):
    cap = cv2.VideoCapture(input_path)
    frame_count = 0
    detected_count = 0
    processed_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % nth == 0:
            #detection_id = f"{os.path.splitext(os.path.basename(input_path))[0]}_frame{frame_count}"
            detection_id = str(uuid.uuid4())
            found = process_frame(frame, detection_id)
            if found:
                detected_count += 1
            processed_count += 1
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {processed_count} frames (every {nth}th), detected {detected_count} license plates out of {frame_count} total frames.")
else:
    # Process as image
    Ivehicle = cv2.imread(input_path)
    if Ivehicle is not None:
        found = process_frame(Ivehicle, detection_id)
        if found:
            print(f"Detected license plate saved for image {input_path}.")
        else:
            print(f"No license plate detected in image {input_path}.")
    else:
        print(f"Failed to read image: {input_path}")