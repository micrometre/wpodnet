# WPODNet-PyTorch Tech Stack

This project is a PyTorch implementation of WPOD-NET (Warped Planar Object Detection Network) for license plate detection. Here's the tech stack used:

## Core Technologies
- **Python**: Main programming language
- **PyTorch**: Deep learning framework for model definition and inference
- **PIL (Pillow)**: Image processing library for loading, saving, and manipulating images
- **OpenCV (cv2)**: Used for video processing and frame extraction
- **NumPy**: Numerical computing library for array operations and matrix manipulations

## Model Architecture
- **WPODNet**: Custom neural network architecture for license plate detection
  - Uses convolutional blocks, residual blocks, and max pooling
  - Outputs probability maps and affine transformation parameters

## Supporting Libraries
- **tqdm**: Progress bar visualization
- **torchvision**: Used for transformations and image processing utilities
- **argparse**: Command-line argument parsing

## Project Summary
This project implements a license plate detection system using the WPOD-NET architecture in PyTorch. The model is capable of:

1. Detecting license plates in images and videos
2. Providing bounding polygon coordinates for detected plates
3. Calculating confidence scores for detections
4. Warping detected plates to a standard rectangular form

The application supports both batch processing of image directories and video files, with features like confidence thresholding, frame skipping for videos, and saving both annotated images and warped license plates.

The implementation includes a clean prediction interface that handles resizing images to appropriate dimensions, performing inference with the model, and post-processing the results to obtain license plate coordinates and confidence scores.





## Quick Run
1. Clone this repository
    ```bash
    git clone https://github.com/Pandede/WPODNet-Pytorch.git
    ```
2. Install [PyTorch](https://pytorch.org) depends on your environment.
3. Install packages in `requirements.txt`
    ```bash
    pip3 install -r requirements.txt
    ```
4. Download the pretrained weight `wpodnet.pth` from [here](https://github.com/Pandede/WPODNet-Pytorch/releases/download/1.0.0/wpodnet.pth)
5. Predict with an image
    ```bash
    python3 predict.py  docs/sample/original/03009.jpg          # The path to the an image
                        # docs/sample/original                  # OR the path to the directory with bulk of images
                        -w weights/wpodnet.pth                  # The path to the weight
                        --save-annotated docs/sample/annotated  # The directory to save the annotated images
                        --save-warped docs/sample/warped        # The directory to save the warped images
    ```


# WPODNet: Build with Torch
## Introduction
This repository implements the proposed method from **ECCV 2018 paper ["License Plate Detection and Recognition in Unconstrained Scenarios"](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sergio_Silva_License_Plate_Detection_ECCV_2018_paper.pdf)** in Torch.

The model in Keras is built by the essay author, see [sergiomsilva/alpr-unconstrained](https://github.com/sergiomsilva/alpr-unconstrained).
