import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

class ConvBlock(torch.nn.Module):
    """Custom convolution block used in the WPODNet architecture"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn_layer = torch.nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.bn_layer(x)
        x = torch.nn.functional.relu(x)
        return x

class ResidualBlock(torch.nn.Module):
    """Residual block with convolutional layers"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = ConvBlock(channels, channels)
        self.sec_layer = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.bn_layer = torch.nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = self.conv_block(x)
        x = self.sec_layer(x)
        x = self.bn_layer(x)
        x += residual
        x = torch.nn.functional.relu(x)
        return x

class WPODNet(torch.nn.Module):
    """Accurate implementation of the WPODNet model for license plate detection"""
    def __init__(self):
        super(WPODNet, self).__init__()
        
        # Backbone architecture matching the pre-trained weights
        self.backbone = torch.nn.ModuleList([
            # Initial blocks
            ConvBlock(3, 16),
            ConvBlock(16, 16),
            torch.nn.MaxPool2d(2, 2),
            ConvBlock(16, 32),
            ResidualBlock(32),
            torch.nn.MaxPool2d(2, 2),
            ConvBlock(32, 64),
            ResidualBlock(64),
            ResidualBlock(64),
            torch.nn.MaxPool2d(2, 2),
            ConvBlock(64, 64),
            ResidualBlock(64),
            ResidualBlock(64),
            torch.nn.MaxPool2d(2, 2),
            ConvBlock(64, 128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        ])
        
        # Detection heads with correct dimensions from the checkpoint
        self.prob_layer = torch.nn.Conv2d(128, 2, 3, padding=1)  # 2 outputs with 3x3 kernel
        self.bbox_layer = torch.nn.Conv2d(128, 6, 3, padding=1)  # 6 outputs with 3x3 kernel
    
    def forward(self, x):
        # Pass input through backbone
        for layer in self.backbone:
            x = layer(x)
        
        # Generate confidence and bounding box predictions
        prob = self.prob_layer(x)
        bbox = self.bbox_layer(x)
        
        return prob, bbox

def preprocess_image(img_path, target_size=(240, 80)):
    """Preprocess image for inference"""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
    
    # Convert to RGB (from BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Store original dimensions
    orig_h, orig_w = img.shape[:2]
    
    # Resize image
    img_resized = cv2.resize(img, (target_size[1], target_size[0]))
    
    # Normalize
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Convert to torch tensor and add batch dimension
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return img, img_tensor, (orig_h, orig_w)

def decode_predictions(prob, bbox, confidence_threshold=0.5):
    """
    Decode model predictions to get license plate bounding boxes
    
    Args:
        prob: Probability output tensor (batch_size, 2, H, W)
        bbox: Bounding box output tensor (batch_size, 6, H, W)
        confidence_threshold: Minimum confidence to consider a detection
    """
    # Get confidence scores - assuming first channel is positive class probability
    confidence = torch.sigmoid(prob[0, 0, :, :])
    
    # Find locations with confidence above threshold
    locations = (confidence > confidence_threshold).nonzero(as_tuple=False)
    
    results = []
    for y, x in locations:
        # Extract bounding box parameters at this location
        # Assuming bbox format is [x_offset, y_offset, w, h, angle_sin, angle_cos]
        bbox_params = bbox[0, :4, y, x].detach().cpu().numpy()
        
        # Convert to x, y, width, height format
        cx, cy = x.item(), y.item()
        w, h = np.exp(bbox_params[2]), np.exp(bbox_params[3])
        x1 = (cx + bbox_params[0] - w/2)
        y1 = (cy + bbox_params[1] - h/2)
        
        score = confidence[y, x].item()
        results.append({
            'bbox': [x1, y1, w, h],
            'confidence': score
        })
    
    return results

def display_results(original_img, detections, orig_dims, model_dims=(240, 80), threshold=0.5):
    """Display image with license plate detections"""
    img_height, img_width = orig_dims
    
    # Create a copy of the image for visualization
    vis_img = original_img.copy()
    
    for det in detections:
        if det['confidence'] < threshold:
            continue
            
        # Scale bbox from model dimensions to original image dimensions
        x, y, w, h = det['bbox']
        x = int(x * img_width / model_dims[1])
        y = int(y * img_height / model_dims[0])
        w = int(w * img_width / model_dims[1])
        h = int(h * img_height / model_dims[0])
        
        # Draw bounding box
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add confidence text
        text = f"{det['confidence']:.2f}"
        cv2.putText(vis_img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the result
    plt.figure(figsize=(10, 8))
    plt.imshow(vis_img)
    plt.axis('off')
    plt.title(f"Detected {len(detections)} license plates")
    plt.show()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='License plate detection using WPODNet')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--weights', type=str, default='weights/wpodnet.pth',
                        help='Path to the model weights (default: weights/wpodnet.pth)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for detection (default: 0.5)')
    parser.add_argument('--save', action='store_true',
                        help='Save the output image instead of displaying it')
    parser.add_argument('--output', type=str, default='output.jpg',
                        help='Path to save the output image (default: output.jpg)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Load model weights
    model_path = Path(args.weights)
    if not model_path.exists():
        print(f"Error: Weights file not found at {model_path}")
        return
    
    # Initialize model
    model = WPODNet()
    
    # Load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Successfully loaded weights from {model_path}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Preprocess image
    try:
        original_img, img_tensor, orig_dims = preprocess_image(args.image_path)
    except Exception as e:
        print(f"Error processing image: {e}")
        return
    
    # Run inference
    with torch.no_grad():
        prob, bbox = model(img_tensor)
    
    # Decode predictions
    detections = decode_predictions(prob, bbox, confidence_threshold=args.threshold)
    
    # Display or save results
    if len(detections) > 0:
        print(f"Found {len(detections)} license plates")
        if args.save:
            img_height, img_width = orig_dims
            vis_img = original_img.copy()
            
            for det in detections:
                if det['confidence'] < args.threshold:
                    continue
                    
                # Scale bbox from model dimensions to original image dimensions
                x, y, w, h = det['bbox']
                x = int(x * img_width / 80)
                y = int(y * img_height / 240)
                w = int(w * img_width / 80)
                h = int(h * img_height / 240)
                
                # Draw bounding box
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add confidence text
                text = f"{det['confidence']:.2f}"
                cv2.putText(vis_img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert back to BGR for OpenCV
            vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(args.output, vis_img_bgr)
            print(f"Saved output image to {args.output}")
        else:
            display_results(original_img, detections, orig_dims, threshold=args.threshold)
    else:
        print("No license plates detected")
        if not args.save:
            plt.imshow(original_img)
            plt.axis('off')
            plt.title("No license plates detected")
            plt.show()

if __name__ == "__main__":
    main()