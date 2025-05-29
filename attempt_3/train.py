import os
import numpy as np # type: ignore
import time
import torch
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader # type: ignore
from torch.optim.lr_scheduler import CosineAnnealingLR  # Add this import
from chess import pgn # type: ignore
from tqdm import tqdm # type: ignore
import pickle

from other_functions import create_input_for_nn, encode_moves_spatial
from dataset_class import ChessDatasetSpatial
from chess_model_class import ChessModelSpatial


def load_pgn(file_path):
    games = []
    with open(file_path, 'r') as pgn_file:
        while True:
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games

files = [file for file in os.listdir("./") if file.endswith(".pgn")]
LIMIT_OF_FILES = min(len(files), 28)

# Estimate number of positions per game (average)
POSITIONS_PER_GAME = 105  # Chess games average about 70 moves
TARGET_POSITIONS = 2000000  # The number of positions we want
# TARGET_POSITIONS = 1500000  # The number of positions we want
ESTIMATED_GAMES_NEEDED = TARGET_POSITIONS // POSITIONS_PER_GAME

# Load only the estimated number of games needed
games = []
positions_loaded = 0
i = 1
for file in tqdm(files, desc="Loading PGN files"):
    print(f"Loading {file}")
    file_games = load_pgn(file)
    for i, game in enumerate(file_games):
        if i % 100 == 0:
            print(f"Loaded {i} games from {file}")
        games.append(game)
        positions_loaded += len(list(game.mainline_moves()))
        if len(games) >= ESTIMATED_GAMES_NEEDED:
            break
    if len(games) >= ESTIMATED_GAMES_NEEDED:
        break
    if i >= LIMIT_OF_FILES:
        break
    i += 1

print(f"GAMES PARSED: {len(games)}")
print(f"Estimated positions: {positions_loaded}")
X, y = create_input_for_nn(games)

print(f"NUMBER OF SAMPLES: {len(y)}")
X = X[0:2500000]
y = y[0:2500000]

# Use spatial encoding instead of categorical
y_spatial = encode_moves_spatial(y)

print(f"Spatial move encoding shape: {y_spatial.shape}")  # Should be (num_samples, 2, 8, 8)

# Save the training data as .npz file
print("Saving training data to spatial_training_data.npz...")
np.savez("spatial_training_data.npz", X=X, y=y_spatial)
print("Training data saved successfully!")

X = torch.tensor(X, dtype=torch.float32)
y_spatial = torch.tensor(y_spatial, dtype=torch.float32)

# Create Dataset and DataLoader with larger batch size
dataset = ChessDatasetSpatial(X, y_spatial)
batch_size = 768  # Increased from 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


class SpatialMoveLoss(nn.Module):
    """Custom loss function for spatial move prediction"""
    
    def __init__(self):
        super(SpatialMoveLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        # predictions: (batch_size, 2, 8, 8) - logits
        # targets: (batch_size, 2, 8, 8) - one-hot spatial representations
        
        batch_size = predictions.size(0)
        
        # Separate from and to predictions
        from_predictions = predictions[:, 0, :, :].view(batch_size, -1)  # (batch_size, 64)
        to_predictions = predictions[:, 1, :, :].view(batch_size, -1)    # (batch_size, 64)
        
        # Separate from and to targets and convert to class indices
        from_targets = targets[:, 0, :, :].view(batch_size, -1)  # (batch_size, 64)
        to_targets = targets[:, 1, :, :].view(batch_size, -1)    # (batch_size, 64)
        
        # Convert one-hot to class indices
        from_target_indices = torch.argmax(from_targets, dim=1)  # (batch_size,)
        to_target_indices = torch.argmax(to_targets, dim=1)      # (batch_size,)
        
        # Calculate losses
        from_loss = self.cross_entropy(from_predictions, from_target_indices)
        to_loss = self.cross_entropy(to_predictions, to_target_indices)
        
        # Combine losses
        total_loss = from_loss + to_loss
        
        return total_loss


# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Model Initialization - no need for num_classes anymore
model = ChessModelSpatial().to(device)
criterion = SpatialMoveLoss()

# Initialize optimizer with slightly higher learning rate for larger batch
initial_lr = 0.001  # Increased from 0.0001
optimizer = optim.Adam(model.parameters(), lr=initial_lr)

# Add cosine annealing scheduler
num_epochs = 250
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

print(f"Training with batch size: {batch_size}, initial learning rate: {initial_lr}")
print(f"Model output shape: (batch_size, 2, 8, 8) - spatial move representation")

for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    
    for inputs, labels in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        running_loss += loss.item()
    
    # Step the scheduler after each epoch
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    
    end_time = time.time()
    epoch_time = end_time - start_time
    minutes: int = int(epoch_time // 60)
    seconds: int = int(epoch_time) - minutes * 60
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}, LR: {current_lr:.6f}, Time: {minutes}m{seconds}s')
    
# Save the model
torch.save(model.state_dict(), "models/trainv3_spatial_model_epochs250.pth")

print("Training completed! Model saved as trainv3_spatial_model_epochs250.pth")
print("This model uses spatial move encoding (2x8x8 channels) instead of categorical encoding.")

