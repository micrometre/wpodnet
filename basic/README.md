# Automatic Vehicle License Plate Detection with WPOD-NET

## Overview
This project provides a Python-based pipeline for automatic vehicle license plate detection using the WPOD-NET (Weakly-supervised Perspective-n-point Object Detection Network) deep learning model. The system processes an input vehicle image, detects the license plate region, and saves both the original and detected license plate images for further use.

## Features
- Loads a pre-trained WPOD-NET model for robust license plate localization.
- Processes vehicle images using OpenCV.
- Detects and extracts license plate regions from input images.
- Displays and saves both the original vehicle image and the detected license plate image.
- Modular code structure for easy extension to license plate recognition (OCR) or batch processing.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Keras (for model loading)
- WPOD-NET model files: `wpod-net.json` (architecture) and `wpod-net.h5` (weights)

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Setup
1. Place your input vehicle images in the `test/` directory (e.g., `test/2.jpg`).
2. Ensure the WPOD-NET model files (`wpod-net.json` and `wpod-net.h5`) are present in the project root.
3. Adjust the `img_path` variable in `main.py` to point to your desired input image.

## Usage
Run the main script:
```bash
python main.py
```

- The script will display the original vehicle image and the detected license plate (if found).
- It will save `original_image.jpg` (the input image) and `detected_license_plate.jpg` (the detected plate) in the project directory.

## Notes
- The detection quality depends on the input image resolution and the WPOD-NET model.
- If no license plate is detected, the output image may be missing or empty.
- The code can be extended to process multiple images or to perform OCR on the detected plates.

## File Structure
- `main.py` — Main script for detection and visualization.
- `local_utils.py` — Utility functions for model loading and image processing.
- `wpod-net.json` — WPOD-NET model architecture.
- `wpod-net.h5` — WPOD-NET model weights.
- `test/` — Directory for input images.

## References
- [WPOD-NET Paper](https://arxiv.org/abs/1804.07017)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Keras Documentation](https://keras.io/) 