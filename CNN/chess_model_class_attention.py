import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class MultiHeadBoardAttention(nn.Module):
    def __init__(self, channels, num_heads=8, head_dim=32, dropout=0.1):
        super(MultiHeadBoardAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        # Projections for Q, K, V
        self.q_proj = nn.Conv2d(channels, num_heads * head_dim, 1)
        self.k_proj = nn.Conv2d(channels, num_heads * head_dim, 1)
        self.v_proj = nn.Conv2d(channels, num_heads * head_dim, 1)
        self.out_proj = nn.Conv2d(num_heads * head_dim, channels, 1)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, _, H, W = x.shape
        
        # Project and reshape to (batch, heads, H*W, head_dim)
        q = self.q_proj(x).view(batch_size, self.num_heads, self.head_dim, H*W).transpose(-2, -1)
        k = self.k_proj(x).view(batch_size, self.num_heads, self.head_dim, H*W).transpose(-2, -1)
        v = self.v_proj(x).view(batch_size, self.num_heads, self.head_dim, H*W).transpose(-2, -1)
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape back to board format
        out = out.transpose(-2, -1).contiguous().view(batch_size, -1, H, W)
        return self.out_proj(out)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super(AttentionBlock, self).__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.attention = MultiHeadBoardAttention(channels, num_heads=num_heads)
        self.norm2 = nn.BatchNorm2d(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 4, channels, 1)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Attention with residual connection
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = residual + x
        
        # MLP with residual connection
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = residual + x
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        out += identity
        out = self.relu(out)
        return out

class ChessModelAttention(nn.Module):
    def __init__(self, num_blocks=8, dropout_rate=0.2):
        super(ChessModelAttention, self).__init__()
        
        # Increased width in the initial layers
        self.input_conv = nn.Sequential(
            nn.Conv2d(13, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Alternating residual and attention blocks
        self.main_tower = nn.ModuleList()
        for i in range(num_blocks):
            self.main_tower.append(ResidualBlock(384))
            self.main_tower.append(AttentionBlock(384, num_heads=8))
        
        # Add dropout after main tower
        self.post_tower_dropout = nn.Dropout2d(dropout_rate)
        
        # Move prediction heads with reduced width for final predictions
        self.from_square_head = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.5),
            nn.Conv2d(128, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
        self.to_square_head = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.5),
            nn.Conv2d(128, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
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
        # Initial convolution with increased width
        x = self.input_conv(x)
        
        # Process through alternating residual and attention blocks
        for block in self.main_tower:
            x = block(x)
        
        # Apply dropout after main tower
        x = self.post_tower_dropout(x)
        
        # Predict from and to squares separately
        from_logits = self.from_square_head(x)
        to_logits = self.to_square_head(x)
        
        # Remove the channel dimension and concatenate
        from_logits = from_logits.squeeze(1)
        to_logits = to_logits.squeeze(1)
        
        # Stack to create (batch_size, 2, 8, 8) output
        output = torch.stack([from_logits, to_logits], dim=1)
        
        return output 