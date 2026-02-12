import torch.nn as nn   
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        # Feature extraction
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=3,
                               padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
    
        # Classification
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Non-linearty added to each layer
        x = self.pool(F.relu(self.conv1(x))) # (B, 32, 112, 112)
        x = self.pool(F.relu(self.conv2(x))) # (B, 64, 56, 56)
        x = self.pool(F.relu(self.conv3(x))) # (B, 128, 28, 28)

        x = x.view(x.size(0), -1) # Flatten, Turning 4D tensor into 2D
        x = F.relu(self.fc1(x)) # Summarize spatial feature
        x = self.dropout(x) # Non-activate 50% neuron from training
        x = self.fc2(x) # Turning 256 feature into outputs

        return x