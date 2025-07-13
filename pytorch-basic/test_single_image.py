import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the classes for CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the same model architecture used for training
class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = torch.nn.Linear(128 * 4 * 4, 512)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(512, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def predict_image(image_path, model, transform):
    """
    Make a prediction on a single image
    """
    image = Image.open(image_path)
    
    # Original image for display
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)
    
    # If image is not 32x32, resize it
    if image.size != (32, 32):
        print(f"Resizing image from {image.size} to (32, 32)")
        image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get model prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        prediction_idx = predicted.item()
        
    # Get probabilities using softmax
    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    
    # Print prediction and confidence
    print(f"Prediction: {classes[prediction_idx]}")
    print(f"Confidence: {probabilities[prediction_idx].item() * 100:.2f}%")
    
    # Show probability distribution
    plt.subplot(1, 2, 2)
    plt.bar(classes, probabilities.cpu().numpy())
    plt.title('Class Probabilities')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return classes[prediction_idx], probabilities[prediction_idx].item()

def main():
    # Define the same transformation as used during training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load the trained model
    model_path = 'cifar_net.pth'
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    
    model = ConvNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Test with some sample images from the CIFAR-10 test set
    # This will download the dataset if not already present
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    
    # Choose a sample image from the test set
    sample_idx = 2  # You can change this to test different images
    sample_image, sample_label = testset[sample_idx]
    
    # Save the sample image to disk
    sample_image_np = sample_image.numpy().transpose((1, 2, 0))
    # Denormalize the image
    sample_image_np = sample_image_np * 0.5 + 0.5
    sample_image_np = (sample_image_np * 255).astype(np.uint8)
    sample_image_pil = Image.fromarray(sample_image_np)
    
    sample_path = '1.jpg'
    sample_image_pil.save(sample_path)
    print(f"Saved sample image to {sample_path}")
    print(f"True label: {classes[sample_label]}")
    
    # Test with the sample image
    print("\nTesting with sample image from CIFAR-10:")
    predict_image(sample_path, model, transform)
    
    # If you have your own test image
    print("\nTo test with your own image, use:")
    print("predict_image('path/to/your/image.jpg', model, transform)")
    
    # Uncomment the next line to test with your own image
    #predict_image('3.jpg', model, transform)

if __name__ == "__main__":
    main()