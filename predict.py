import errno
from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path
from typing import List, Union
import os
import cv2
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import easyocr
import numpy as np
import uuid
import json
import datetime
from collections import defaultdict
import requests

from wpodnet import Predictor, load_wpodnet_from_checkpoint



# create images directory if it does not existif 

if not os.path.exists("images"):
    os.makedirs("images")
if not os.path.exists("images/annotated"):
    os.makedirs("images/annotated")
if not os.path.exists("images/warped"):
    os.makedirs("images/warped")        
if not os.path.exists("images/original"):
    os.makedirs("original")
# Ensure the current directory is in the Python path

# Initialize EasyOCR reader (will be done once)
def get_ocr_reader(languages=['en']):
    """
    Initialize and return an EasyOCR reader.
    
    Args:
        languages (list): List of language codes to use
    
    Returns:
        easyocr.Reader: Initialized EasyOCR reader
    """
    try:
        return easyocr.Reader(languages)
    except Exception as e:
        print(f"Error initializing EasyOCR: {e}")
        print("OCR functionality will be disabled")
        return None


def list_image_paths(p: Union[str, Path]) -> List[Path]:
    """
    List all images in a directory.

    Args:
        path (Union[str, Path]): The path to the directory containing images.

    Returns:
        Generator[Image.Image]: A generator of PIL Image objects.
    """
    p = Path(p)
    if not p.is_dir():
        raise FileNotFoundError(errno.ENOTDIR, "No such directory", args.save_annotated)

    image_paths: List[Path] = []
    for f in p.glob("**/*"):
        try:
            with Image.open(f) as image:
                image.verify()
            image_paths.append(f)
        except UnidentifiedImageError:
            pass
    return image_paths


def is_video_file(file_path: Union[str, Path]) -> bool:
    """
    Check if a file is a video file by extension.

    Args:
        file_path (Union[str, Path]): The path to the file.

    Returns:
        bool: True if the file is a video file, False otherwise.
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    return Path(file_path).suffix.lower() in video_extensions


def extract_text_from_plate(plate_image, ocr_reader):
    """
    Extract text from a license plate image using EasyOCR.
    
    Args:
        plate_image (PIL.Image): The warped license plate image
        ocr_reader: EasyOCR reader instance
    
    Returns:
        str: Extracted license plate text or empty string if none found
    """
    if ocr_reader is None:
        return ""
        
    # Convert PIL image to numpy array for EasyOCR
    img_array = np.array(plate_image)
    
    try:
        # Perform OCR
        results = ocr_reader.readtext(img_array)
        
        # Extract and combine text results
        if results:
            texts = [text for _, text, conf in results]
            # Convert to uppercase
            return " ".join(texts).upper()
        else:
            return ""
    except Exception as e:
        print(f"OCR error: {e}")
        return ""


def process_video(
    video_path: Path, 
    predictor: Predictor, 
    scaling_ratio: float = 1.0,
    save_annotated: Union[Path, None] = None,
    save_warped: Union[Path, None] = None,
    frame_skip: int = 1,
    confidence_threshold: float = 0.5,
    ocr_reader = None,
    detection_results = None,
    server_url = None  # Add server_url parameter
) -> None:
    """
    Process a video file frame by frame.

    Args:
        video_path (Path): The path to the video file.
        predictor (Predictor): The license plate predictor.
        scaling_ratio (float, optional): The scaling ratio. Defaults to 1.0.
        save_annotated (Union[Path, None], optional): Directory to save annotated frames. Defaults to None.
        save_warped (Union[Path, None], optional): Directory to save warped license plates. Defaults to None.
        frame_skip (int, optional): Process every nth frame. Defaults to 1.
        confidence_threshold (float, optional): Minimum confidence score. Defaults to 0.5.
        ocr_reader: EasyOCR reader instance for text extraction.
        detection_results: Dictionary to store detection results
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    print(f"Processing video: {video_path.name}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    
    video_basename = video_path.stem
    
    # Process frames with progress bar
    with tqdm(total=total_frames // frame_skip) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every nth frame
            if frame_count % frame_skip == 0:
                # Convert OpenCV BGR to RGB for PIL
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                # Run detection
                prediction = predictor.predict(pil_image, scaling_ratio=scaling_ratio)
                
                # If confidence is high enough
                if prediction.confidence >= confidence_threshold:
                    # Generate unique ID for this detection
                    detection_id = str(uuid.uuid4())
                    
                    # Get warped license plate image
                    warped = prediction.warp(pil_image)
                    
                    # Extract text using OCR if available
                    plate_text = ""
                    if ocr_reader:
                        plate_text = extract_text_from_plate(warped, ocr_reader)
                        plate_text = plate_text.upper()  # Convert to uppercase
                        # Post-processing for license plates
                        plate_text = plate_text.replace('I', '1')
                        
                    print(f"\nFrame {frame_count}: License plate detected")
                    print(f"  Confidence: {prediction.confidence:.4f}")
                    if plate_text:
                        print(f"  Plate text: {plate_text}")
                    print(f"  ID: {detection_id}")
                    
                    # Save annotated frame if requested
                    if save_annotated:
                        annotated = pil_image.copy()
                        prediction.annotate(annotated, outline="red")
                        
                        # Add text to the annotated image if available
                        if plate_text:
                            draw = ImageDraw.Draw(annotated)
                            bbox = prediction.bounds
                            text_pos = (bbox[0][0], bbox[0][1] - 20)  # Position above the plate
                            # Draw with contrasting background for readability
                            text_bbox = draw.textbbox(text_pos, plate_text)
                            draw.rectangle(text_bbox, fill="black")
                            draw.text(text_pos, plate_text, fill="white")
                            
                        output_path = save_annotated / f"{video_basename}_frame_{frame_count}_{detection_id}.jpg"
                        annotated.save(output_path)
                        
                    # Save warped license plate if requested
                    if save_warped:
                        output_path = save_warped / f"{video_basename}_frame_{frame_count}_{detection_id}_plate.jpg"
                        warped.save(output_path)
                        
                        # Add to detection results
                        if detection_results is not None:
                            orig_frame_path = ""
                            if save_annotated:
                                orig_frame_path = str(save_annotated / f"{video_basename}_frame_{frame_count}_{detection_id}.jpg")
                                
                            detection_results[detection_id].append((
                                plate_text,
                                str(output_path),
                                orig_frame_path
                            ))
                            
                            # Send plate info via HTTP POST if server URL is provided
                            if server_url:
                                # Format the data similar to main.py
                                plate_info = {
                                    "uuid": detection_id,
                                    "results": [
                                        {
                                            "plate": plate_text,
                                            "image_path": str(output_path),
                                            "original_frame_path": orig_frame_path
                                        }
                                    ]
                                }
                                
                                # Send HTTP POST request
                                headers = {"Content-Type": "application/json"}
                                try:
                                    response = requests.post(server_url, json=plate_info, headers=headers)
                                    if response.status_code == 200:
                                        print(f"  Data successfully sent to server: {response.json()}")
                                    else:
                                        print(f"  Failed to send data. Status code: {response.status_code}")
                                except Exception as e:
                                    print(f"  Error occurred while sending data: {e}")
                        
                pbar.update(1)
            
            frame_count += 1
            
    cap.release()
    print(f"Video processing complete: {frame_count // frame_skip} frames processed")


def save_detection_results_to_json(results_dict, output_dir=None):
    """
    Save detection results to a JSON file.
    
    Args:
        results_dict (dict): Dictionary of detection results organized by UUID
        output_dir (Path, optional): Directory to save JSON file. Defaults to outputs directory.
    
    Returns:
        Path: Path to the saved JSON file
    """
    # Format for JSON output
    output_list = [
        {
            "uuid": uuid,
            "results": [
                {
                    "plate": plate_text,
                    "image_path": img_path,
                    "original_frame_path": orig_path
                } for plate_text, img_path, orig_path in detections
            ]
        } for uuid, detections in results_dict.items()
    ]
    
    # Create timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detected_plates_{timestamp}.json"
    
    # Use outputs directory by default if no output_dir specified
    if output_dir:
        output_path = Path(output_dir) / filename
    else:
        output_path = Path("outputs") / filename
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to JSON file with indentation for readability
    with open(output_path, 'w') as f:
        json.dump(output_list, f, indent=4)
    
    print(f"Detection results saved to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("source", type=str, help="the path to the video file")
    parser.add_argument(
        "-w", "--weight", 
        type=str, required=True,
        default="weights/wpodnet.pth",
        help="the path to the model weight"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="adjust the scaling ratio. default to 1.0.",
    )
    parser.add_argument(
        "--save-annotated",
        type=str,
        default="images/annotated",
        help="save the annotated frames at the given folder",
    )
    parser.add_argument(
        "--save-warped", 
        type=str, 
        default="images/warped",
        help="save the warped license plates at the given folder"
    )
    # Video-specific arguments
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=25,
        help="process every nth frame in videos. default to 1 (process all frames)"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.4,
        help="minimum confidence score for a detection to be considered valid. default to 0.5"
    )
    # OCR-related arguments
    parser.add_argument(
        "--ocr-language",
        type=str,
        default="en",
        help="language for OCR (comma-separated list for multiple languages, e.g. 'en,es')"
    )
    parser.add_argument(
        "--disable-ocr",
        action="store_true",
        help="disable OCR text extraction"
    )
    # Add new argument for JSON output
    parser.add_argument(
        "--save-json",
        type=str,
        default="outputs",
        help="save detection results as JSON to the specified directory (default: outputs)"
    )
    # Server URL argument
    parser.add_argument(
        "--server-url",
        default="http://localhost:8080/alprd",
        type=str,
        help="URL to send plate detection results via HTTP POST (e.g., http://localhost:8080/alprd)"
    )
    
    args = parser.parse_args()

    if args.scale <= 0.0:
        raise ArgumentTypeError(message="scale must be greater than 0.0")
    
    if args.frame_skip <= 0:
        raise ArgumentTypeError(message="frame-skip must be greater than 0")
        
    if args.confidence_threshold < 0.0 or args.confidence_threshold > 1.0:
        raise ArgumentTypeError(message="confidence-threshold must be between 0.0 and 1.0")

    if args.save_annotated is not None:
        save_annotated = Path(args.save_annotated)
        if not save_annotated.is_dir():
            raise FileNotFoundError(
                errno.ENOTDIR, "No such directory", args.save_annotated
            )
    else:
        save_annotated = None

    if args.save_warped is not None:
        save_warped = Path(args.save_warped)
        if not save_warped.is_dir():
            raise FileNotFoundError(
                errno.ENOTDIR, "No such directory", args.save_warped
            )
    else:
        save_warped = None

    # Initialize OCR if not disabled
    ocr_reader = None
    if not args.disable_ocr:
        # Parse languages list
        languages = [lang.strip() for lang in args.ocr_language.split(',')]
        print(f"Initializing EasyOCR with languages: {languages}")
        from PIL import ImageDraw
        ocr_reader = get_ocr_reader(languages)

    # Prepare for the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = load_wpodnet_from_checkpoint(args.weight).to(device)

    predictor = Predictor(model)

    source = Path(args.source)
    # Dictionary to store detection results
    detection_results = defaultdict(list)
    
    # Check if source is a video file
    if not source.is_file() or not is_video_file(source):
        raise ValueError(f"Source must be a video file. Got: {source}")
        
    # Process video file
    process_video(
        source, 
        predictor, 
        scaling_ratio=args.scale,
        save_annotated=save_annotated,
        save_warped=save_warped,
        frame_skip=args.frame_skip,
        confidence_threshold=args.confidence_threshold,
        ocr_reader=ocr_reader,
        detection_results=detection_results,
        server_url=args.server_url  # Pass the server URL
    )
    
    # Save the results if requested
    if args.save_json:
        json_dir = Path(args.save_json)
        json_dir.mkdir(parents=True, exist_ok=True)
        save_detection_results_to_json(detection_results, json_dir)
