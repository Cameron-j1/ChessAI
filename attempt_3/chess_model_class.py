import torch.nn as nn
import torch
import math

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        return out

class ChessModelSpatial(nn.Module):
    def __init__(self, num_residual_blocks=8):
        super(ChessModelSpatial, self).__init__()
        
        # Initial input processing
        self.input_conv = nn.Sequential(
            nn.Conv2d(13, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Residual tower
        self.residual_tower = nn.ModuleList([
            ResidualBlock(256) for _ in range(num_residual_blocks)
        ])
        
        # Move prediction heads - separate for from and to squares
        self.from_square_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)  # Output 1 channel for from squares
        )
        
        self.to_square_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)  # Output 1 channel for to squares
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial convolution
        x = self.input_conv(x)
        
        # Residual blocks
        for block in self.residual_tower:
            x = block(x)
        
        # Predict from and to squares separately
        from_logits = self.from_square_head(x)  # (batch_size, 1, 8, 8)
        to_logits = self.to_square_head(x)      # (batch_size, 1, 8, 8)
        
        # Remove the channel dimension and concatenate
        from_logits = from_logits.squeeze(1)    # (batch_size, 8, 8)
        to_logits = to_logits.squeeze(1)        # (batch_size, 8, 8)
        
        # Stack to create (batch_size, 2, 8, 8) output
        output = torch.stack([from_logits, to_logits], dim=1)
        
        return output

# Keep the old model for backward compatibility
class ChessModel(nn.Module):
    def __init__(self, num_classes, num_residual_blocks=8):
        super(ChessModel, self).__init__()
        
        # Initial input processing
        self.input_conv = nn.Sequential(
            nn.Conv2d(13, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Residual tower
        self.residual_tower = nn.ModuleList([
            ResidualBlock(256) for _ in range(num_residual_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial convolution
        x = self.input_conv(x)
        
        # Residual blocks
        for block in self.residual_tower:
            x = block(x)
        
        # Policy head
        x = self.policy_conv(x)
        
        # Final fully connected layers
        x = self.fc_layers(x)
        
        return x