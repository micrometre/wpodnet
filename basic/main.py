import cv2
import numpy as np
import tensorflow as tf
from local_utils import (get_plate, lphm_rotation, transform_plate, 
                   decode_plate, draw_plate, load_wpod_net)

# Load the pretrained WPOD-NET model
model_path = 'wpod-net.json'    # Model architecture file
weights_path = 'wpod-net.h5'    # Model weights file
wpod_net = load_wpod_net(model_path, weights_path)

# Load an image
image_path = 'test/1.jpg'
image = cv2.imread(image_path)

# Preprocess the image for WPOD-NET
img_resized = cv2.resize(image, (512, 512))
img_resized = img_resized.astype('float32')
img_resized /= 255.0
img_resized = np.expand_dims(img_resized, axis=0)

# Run detection
predictions = wpod_net.predict(img_resized)

# Process predictions to get plate coordinates
plates = get_plate(predictions, img_resized.shape)

# Draw detected plates on the original image
for plate in plates:
    # Get plate corners
    corners = plate['corners']
    
    # Draw the plate polygon
    image = draw_plate(image, corners)
    
    # Optional: Extract and rotate the plate
    plate_img = transform_plate(image, corners)
    
    # If you have OCR, you could process plate_img here

# Save or display the result
cv2.imwrite('detected_plate.jpg', image)
cv2.imshow('License Plate Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()