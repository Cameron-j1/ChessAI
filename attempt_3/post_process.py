import chess
import torch
import numpy as np
import pickle
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random
from tqdm import tqdm
from chess_model_class import ChessModel
from other_functions import board_to_matrix
from dataset_class import ChessDataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import multiprocessing
from functools import partial

# Set start method to 'spawn' for CUDA compatibility
multiprocessing.set_start_method('spawn', force=True)

BATCH_SIZE = 64  # Batch size for GPU processing

class Node:
    def __init__(self, board: chess.Board, parent=None, move=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move  # Move that led to this node
        self.children: Dict[str, Node] = {}
        self.visits = 0
        self.value = 0.0
        self.is_terminal = board.is_game_over()
        
    def add_child(self, move: chess.Move) -> 'Node':
        board_copy = self.board.copy()
        board_copy.push(move)
        self.children[move.uci()] = Node(board_copy, self, move)
        return self.children[move.uci()]

    def get_untried_moves(self) -> List[chess.Move]:
        return [move for move in self.board.legal_moves 
                if move.uci() not in self.children]

class MCTS:
    def __init__(self, model, move_to_int, device, exploration_weight=1.0):
        self.model = model
        self.move_to_int = move_to_int
        self.device = device
        self.exploration_weight = exploration_weight
        self.policy_cache = {}  # Cache for model predictions
        self.cache_size = 10000  # Reduced from 10000 to 1000
        self.batch_buffer = []  # Buffer for batch processing
        self.batch_fens = []  # Corresponding FENs for batch processing
        
    def clear_cache(self):
        """Clear the policy cache to free memory"""
        self.policy_cache.clear()
        # Force CUDA to release memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process_batch(self):
        if not self.batch_buffer:
            return
            
        # Convert batch to tensor - stack will create a batch dimension automatically
        batch_tensor = torch.stack(self.batch_buffer).to(self.device)  # Shape: [batch_size, 13, 8, 8]
        
        # Get predictions for entire batch
        with torch.no_grad():
            logits = self.model(batch_tensor)
            probabilities = torch.softmax(logits, dim=1)
        
        # Process each board's predictions and update cache
        for i, fen in enumerate(self.batch_fens):
            probs = probabilities[i].cpu().numpy()
            
            policy = {}
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            legal_moves_uci = [move.uci() for move in legal_moves]
            
            for move_uci in legal_moves_uci:
                if move_uci in self.move_to_int:
                    policy[move_uci] = float(probs[self.move_to_int[move_uci]])
                else:
                    policy[move_uci] = 1e-8
                    
            # Normalize
            total = sum(policy.values())
            if total > 0:
                policy = {k: v/total for k, v in policy.items()}
                
            # Cache the result
            if len(self.policy_cache) >= self.cache_size:
                # Clear half the cache when full
                old_keys = list(self.policy_cache.keys())[:self.cache_size//2]
                for k in old_keys:
                    del self.policy_cache[k]
            self.policy_cache[fen] = policy
            
        # Clear buffers and GPU memory
        self.batch_buffer = []
        self.batch_fens = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_model_policy(self, board: chess.Board) -> Dict[str, float]:
        fen = board.fen()
        if fen in self.policy_cache:
            return self.policy_cache[fen]
            
        # Add to batch buffer
        matrix = board_to_matrix(board)  # Shape: [13, 8, 8]
        X_tensor = torch.tensor(matrix, dtype=torch.float32)  # Don't add extra dimension
        self.batch_buffer.append(X_tensor)
        self.batch_fens.append(fen)
        
        # Process batch if buffer is full
        if len(self.batch_buffer) >= BATCH_SIZE:
            self.process_batch()
            
        # Always process the batch if this position isn't cached
        if len(self.batch_buffer) > 0:
            self.process_batch()
            
        # Now the policy should be in cache
        if fen not in self.policy_cache:
            # Fallback to uniform policy if something went wrong
            empty_policy = {move.uci(): 1.0/len(list(board.legal_moves)) 
                          for move in board.legal_moves}
            self.policy_cache[fen] = empty_policy
            
        return self.policy_cache[fen]

    def select(self, node: Node) -> Node:
        while not node.is_terminal:
            if node.get_untried_moves():
                return node
            node = self.get_best_child(node)
        return node

    def expand(self, node: Node) -> Node:
        untried_moves = node.get_untried_moves()
        if not untried_moves:
            return node
        
        # Use model's policy to weight move selection
        policy = self.get_model_policy(node.board)
        untried_moves_uci = [move.uci() for move in untried_moves]
        
        # Filter policy for untried moves
        untried_policy = {move: policy.get(move, 1e-8) for move in untried_moves_uci}
        total = sum(untried_policy.values())
        
        if total > 0:
            probabilities = [untried_policy[move]/total for move in untried_moves_uci]
            move = untried_moves[np.random.choice(len(untried_moves), p=probabilities)]
        else:
            move = random.choice(untried_moves)
            
        return node.add_child(move)

    def simulate(self, node: Node) -> float:
        board = node.board.copy()
        depth = 0
        max_depth = 50  # Prevent infinite games
        
        while not board.is_game_over() and depth < max_depth:
            policy = self.get_model_policy(board)
            legal_moves_uci = [move.uci() for move in board.legal_moves]
            
            # Filter policy for legal moves
            legal_policy = {move: policy.get(move, 1e-8) for move in legal_moves_uci}
            total = sum(legal_policy.values())
            
            if total > 0:
                move_uci = random.choices(
                    list(legal_policy.keys()),
                    weights=list(legal_policy.values())
                )[0]
            else:
                move_uci = random.choice(legal_moves_uci)
                
            board.push_uci(move_uci)
            depth += 1
            
        # Process any remaining items in batch buffer
        if self.batch_buffer:
            self.process_batch()
            
        # Evaluate final position
        outcome = board.outcome()
        if outcome is None:
            return 0.0  # Draw due to max depth
        elif outcome.winner is None:
            return 0.0  # Draw
        else:
            # Return 1 for win, -1 for loss from the perspective of the player who made the last move
            return -1.0 if outcome.winner == board.turn else 1.0

    def backpropagate(self, node: Node, value: float) -> None:
        while node is not None:
            node.visits += 1
            # Flip value because we alternate perspectives
            value = -value
            node.value += value
            node = node.parent

    def get_best_child(self, node: Node) -> Node:
        if not node.children:
            raise ValueError("Node has no children")
            
        def ucb_score(child: Node) -> float:
            if child.visits == 0:
                return float('inf')
            exploitation = child.value / child.visits
            exploration = math.sqrt(math.log(node.visits) / child.visits)
            return exploitation + self.exploration_weight * exploration
            
        return max(node.children.values(), key=ucb_score)

    def search(self, board: chess.Board, num_simulations: int = 100) -> chess.Move:
        root = Node(board)
        
        for _ in range(num_simulations):
            leaf = self.select(root)
            if not leaf.is_terminal:
                leaf = self.expand(leaf)
                value = self.simulate(leaf)
                self.backpropagate(leaf, value)
            
            # Process any remaining items in batch buffer
            if self.batch_buffer:
                self.process_batch()
        
        # Select move with most visits
        best_move = max(root.children.items(), key=lambda x: x[1].visits)[0]
        return chess.Move.from_uci(best_move)

def self_play_game(mcts: MCTS, temperature: float = 1.0) -> List[Tuple[np.ndarray, str]]:
    board = chess.Board()
    training_data = []
    
    move_count = 0
    max_moves = 150  # Maximum number of moves before forcing a draw
    
    # Temperature annealing
    def get_temperature(move_count):
        if move_count < 30:  # Early game
            return 1.0
        elif move_count < 50:  # Mid game
            return 0.5
        else:  # Late game
            return 0.2
    
    while not board.is_game_over() and move_count < max_moves:
        move_count += 1
        if move_count % 10 == 0:  # Clear cache periodically
            mcts.clear_cache()
        print(f"Move {move_count}")
        current_temp = get_temperature(move_count)
        
        # Store position
        matrix = board_to_matrix(board)
        
        # Get move using MCTS with increased simulations
        move = mcts.search(board, num_simulations=100)
        training_data.append((matrix, move.uci()))
        
        # Make move
        board.push(move)
        
    return training_data

def train_on_self_play_data(model: ChessModel, 
                           X: np.ndarray, 
                           y: np.ndarray,
                           device: torch.device,
                           original_X: np.ndarray = None,  # Original training data
                           original_y: np.ndarray = None,
                           num_epochs: int = 50,
                           batch_size: int = 64,
                           learning_rate: float = 0.00001):  # Reduced learning rate
    
    # Mix in some original training data if provided
    if original_X is not None and original_y is not None:
        # Take 20% of original data
        indices = np.random.choice(len(original_X), size=len(original_X)//5, replace=False)
        X = np.concatenate([X, original_X[indices]])
        y = np.concatenate([y, original_y[indices]])
    
    # Split into training and validation sets
    val_size = int(0.1 * len(X))
    indices = np.random.permutation(len(X))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # Convert numpy arrays to tensors with explicit dtypes
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    train_dataset = ChessDataset(X[train_indices], y[train_indices])
    val_dataset = ChessDataset(X[val_indices], y[val_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # Ensure model is in float32 mode
    model = model.type(torch.float32)
    model.train()
    
    for epoch in range(num_epochs):
        # Training phase
        running_loss = 0.0
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
            
        train_loss = running_loss/len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, dtype=torch.float32)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss = val_loss/len(val_loader)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "models/self_play_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break

def generate_single_game(model_path: str, move_to_int: Dict[str, int], device_str: str, game_num: int) -> List[Tuple[np.ndarray, str]]:
    # Set up model for this process
    device = torch.device(device_str)
    model = ChessModel(num_classes=len(move_to_int))
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # Initialize MCTS with higher exploration weight to compensate for fewer simulations
    mcts = MCTS(model, move_to_int, device, exploration_weight=1.4)
    
    print(f"\nGenerating game {game_num}")
    return self_play_game(mcts)

def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Load the original model and move mappings
    with open("models/move_to_int_architecture_v2", "rb") as f:
        move_to_int = pickle.load(f)
    
    # Load original training data
    try:
        original_data = np.load("original_training_data.npz")
        original_X = original_data["X"]
        original_y = original_data["y"]
        print(f"Loaded {len(original_X)} original training positions")
    except:
        print("No original training data found, proceeding without it")
        original_X = None
        original_y = None
    
    model = ChessModel(num_classes=len(move_to_int))
    modelstr = "models/model_architecture_v2_epochs20.pth"
    model.load_state_dict(torch.load(modelstr))
    print(f"Loaded model from {modelstr}")
    model.to(device)
    model.eval()
    
    # Initialize parallel processing
    num_games = 36
    
    # Adjust number of processes based on GPU memory - now more conservative estimate
    total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
    # gpu_memory_per_process = 1.5 * 1024 * 1024 * 1024  # Reduced to 1.5GB per process
    gpu_memory_per_process = 0.3 * 1024 * 1024 * 1024  # Reduced to 1.5GB per process
    num_processes = min(12, int(total_gpu_memory / gpu_memory_per_process))  # Cap at 12 processes
    
    print(f"Generating {num_games} self-play games using {num_processes} parallel processes...")
    print(f"GPU Memory available: {total_gpu_memory / (1024*1024*1024):.2f}GB")
    print(f"Estimated memory per process: {gpu_memory_per_process / (1024*1024*1024):.2f}GB")
    
    # Create a pool of workers
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Prepare the worker function with fixed arguments
        worker_func = partial(
            generate_single_game,
            "models/model_architecture_v2_epochs20.pth",
            move_to_int,
            str(device)
        )
        
        # Run games in parallel and collect results
        all_training_data = []
        for game_data in tqdm(
            pool.imap_unordered(worker_func, range(num_games)),
            total=num_games,
            desc="Self-play games",
            unit="game"
        ):
            all_training_data.extend(game_data)
    
    # Prepare training data
    filtered_data = [(data[0], data[1]) for data in all_training_data 
                    if data[1] in move_to_int]
    X = np.array([data[0] for data in filtered_data], dtype=np.float32)  # Explicitly set dtype
    y = np.array([move_to_int[data[1]] for data in filtered_data], dtype=np.int64)
    
    print(f"Generated {len(y)} valid training positions out of {len(all_training_data)} total positions")
    
    # Train on new data
    train_on_self_play_data(model, X, y, device, original_X, original_y)
    
    # Load and save the best model from training
    model.load_state_dict(torch.load("models/self_play_best.pth"))
    torch.save(model.state_dict(), "models/self_play_improved.pth")
    print("Saved improved model to models/self_play_improved.pth")

if __name__ == "__main__":
    main() 