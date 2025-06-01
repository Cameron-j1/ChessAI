import os
import numpy as np # type: ignore
import time
import torch
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader # type: ignore
from torch.optim.lr_scheduler import CosineAnnealingLR
from chess import pgn # type: ignore
from tqdm import tqdm # type: ignore
import pickle
import gc

from other_functions import create_input_for_nn, encode_moves_spatial
from dataset_class import ChessDatasetSpatial
from chess_model_class_attention import ChessModelAttention  #this import source file will need to change for each different model net you want to train


def load_pgn(file_path):
    games = []
    with open(file_path, 'r') as pgn_file:
        while True:
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games


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


def train_attention_model():
    """Train the attention-enhanced chess model with spatial move prediction"""
    print("Setting up training for attention-enhanced model...")
    
    # Check if npz directory exists and look for spatial training data files
    npz_dir = "./npz"
    
    if os.path.exists(npz_dir):
        print(f"Looking for training data files in {npz_dir}...")
        npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
        
        if npz_files:
            print(f"Found {len(npz_files)} .npz files")
            X_list = []
            y_spatial_list = []
            
            # Load and convert one file at a time
            for npz_file in npz_files:
                file_path = os.path.join(npz_dir, npz_file)
                print(f"Loading data from {npz_file}...")
                
                # Load the existing data
                data = np.load(file_path)
                print(f"Loaded from {npz_file}:")
                print(f"X shape: {data['X'].shape}")
                print(f"y_spatial shape: {data['y'].shape}")
                
                # Convert to torch tensors immediately to avoid keeping both numpy and torch versions
                X_tensor = torch.tensor(data['X'], dtype=torch.float32)
                y_tensor = torch.tensor(data['y'], dtype=torch.float32)
                
                # Clear numpy arrays from memory
                del data
                gc.collect()
                
                X_list.append(X_tensor)
                y_spatial_list.append(y_tensor)
                print(f"Converted {npz_file} to PyTorch tensors")
                
            # Combine all loaded data
            print("\nConcatenating tensors...")
            X = torch.cat(X_list, dim=0)
            y_spatial = torch.cat(y_spatial_list, dim=0)
            
            # Clear individual tensors
            del X_list
            del y_spatial_list
            gc.collect()
            
            print("\nCombined dataset:")
            print(f"Total samples: {len(X)}")
            print(f"Final X shape: {X.shape}")
            print(f"Final y_spatial shape: {y_spatial.shape}")
            
            # Limit dataset to 3.9M positions due to RAM constraints
            max_positions = 3900000
            if len(X) > max_positions:
                print(f"Limiting dataset from {len(X)} to {max_positions} positions due to RAM constraints")
                X = X[:max_positions]
                y_spatial = y_spatial[:max_positions]
            
            print(f"Final dataset size after limiting: {len(X)} positions")
            
        else:
            print("No .npz files found in the npz directory. Processing PGN files...")
            
            # Load PGN files similar to train.py
            files = [file for file in os.listdir("./") if file.endswith(".pgn")]
            LIMIT_OF_FILES = min(len(files), 28)
            print(f'number of files: {LIMIT_OF_FILES}')

            # Load all games from available files
            games = []
            positions_loaded = 0
            for file_idx, file in enumerate(tqdm(files[:LIMIT_OF_FILES], desc="Loading PGN files")):
                print(f"Loading {file}")
                file_games = load_pgn(file)
                for game in file_games:
                    games.append(game)
                    positions_loaded += len(list(game.mainline_moves()))
                print(f"Processed file {file_idx + 1}/{LIMIT_OF_FILES}: {file} - Total games so far: {len(games)}")

            print(f"GAMES PARSED: {len(games)}")
            print(f"Estimated positions: {positions_loaded}")
            
            # Create input data using the same approach as train.py
            X, y = create_input_for_nn(games)
            print(f"NUMBER OF SAMPLES: {len(y)}")
            
            # Use spatial encoding instead of categorical
            y_spatial = encode_moves_spatial(y)
            print(f"Spatial move encoding shape: {y_spatial.shape}")
            
            # Create npz directory if it doesn't exist
            os.makedirs(npz_dir, exist_ok=True)
            
            # Save the training data as .npz file in the npz directory
            spatial_data_file = os.path.join(npz_dir, f"spatial_training_data_new.npz")
            print(f"Saving training data to {spatial_data_file}...")
            np.savez(spatial_data_file, X=X, y=y_spatial)
            print("Training data saved successfully!")
            
            # Convert numpy arrays to torch tensors for PGN-processed data
            X = torch.tensor(X, dtype=torch.float32)
            y_spatial = torch.tensor(y_spatial, dtype=torch.float32)
    else:
        print("No npz directory found, exiting")
        exit()

    # Create Dataset and DataLoader
    dataset = ChessDatasetSpatial(X, y_spatial)
    
    batch_size = 256
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Initialize the attention model
    model = ChessModelAttention(num_blocks=8, dropout_rate=0.2).to(device)
    criterion = SpatialMoveLoss()
    
    initial_lr = 0.0005
    weight_decay = 2e-4  # Increased from 1e-4 for better regularization and to prevent overfitting
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    
    num_epochs = 100
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    print(f"Training attention model with:")
    print(f"- Batch size: {batch_size}")
    print(f"- Initial learning rate: {initial_lr}")
    print(f"- Weight decay: {weight_decay}")
    print(f"- Dropout rate: 0.2")
    print(f"- Number of blocks: 8")
    print(f"- Model width: 384 channels")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        try:
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                running_loss += loss.item()
                batch_count += 1
                
                # Update progress bar
                if batch_count % 10 == 0:
                    progress_bar.set_postfix({'Loss': f'{running_loss/batch_count:.4f}'})
                
                # Clear some memory every 100 batches
                if batch_count % 100 == 0:
                    del outputs, loss
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
        
        # Step the scheduler after each epoch
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        end_time = time.time()
        epoch_time = end_time - start_time
        minutes = int(epoch_time // 60)
        seconds = int(epoch_time) - minutes * 60
        avg_loss = running_loss / batch_count if batch_count > 0 else 0
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, Time: {minutes}m{seconds}s')
        
        # Save checkpoint every epoch for the first few epochs, then every 5
        if epoch < 2 or (epoch + 1) % 5 == 0:
            checkpoint_path = f"models/attention_model_checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save the final model
    final_model_path = "models/v1_attention_model_100epochs.pth"
    torch.save(model.state_dict(), final_model_path)
    
    print(f"Training completed! Final model saved as {final_model_path}")
    print("This model uses attention mechanisms and spatial move encoding.")


if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    train_attention_model() 