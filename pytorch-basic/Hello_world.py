import torch
import torch.nn as nn
import torch.optim as optim

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a simple dataset (XOR problem)
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32).to(device)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32).to(device)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(2, 8)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

# Initialize the network and optimizer
model = SimpleNN().to(device)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
epochs = 10000
for epoch in range(epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    predicted = model(X)
    print("\nInput -> Output:")
    for i in range(4):
        print(f"{X[i].cpu().numpy()} -> {predicted[i].item():.4f} (Expected: {y[i].item()})")

# Demonstration of basic tensor operations
print("\nBasic Tensor Operations:")
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print(f"a + b = {a + b}")
print(f"a * b = {a * b}")
print(f"Matrix multiplication: {torch.matmul(a, b)}")