"""
Enhanced DQN trainer for the SimpleChessEnv.
Uses a complete board representation with convolutional layers for better spatial understanding.
ε-greedy schedule starts by delegating to a pre-trained ChessMoveEvaluator,
then gradually shifts to the neural-net policy.
Enhanced with detailed time diagnostics.
"""

import random, time, math, pickle, gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import chess

from simple_chess_env import SimpleChessEnv
from chess_svm_model import ChessMoveEvaluator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Enhanced Board Representation ----------
def board_to_planes(board):
    """
    Convert a chess board to a set of 12 planes (6 piece types × 2 colors)
    plus additional planes for state information.
    
    Returns a (14, 8, 8) tensor representing the board state.
    """
    # Initialize 14 planes (12 for pieces, 1 for turn, 1 for castling rights)
    planes = np.zeros((14, 8, 8), dtype=np.float32)
    
    # Fill piece planes
    piece_idx = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            planes[piece_idx[piece.symbol()]][row][col] = 1
    
    # Fill turn plane (plane 12)
    if board.turn == chess.WHITE:
        planes[12].fill(1)
    
    # Fill castling rights (plane 13)
    castling_value = 0
    if board.has_kingside_castling_rights(chess.WHITE): castling_value += 1
    if board.has_queenside_castling_rights(chess.WHITE): castling_value += 2
    if board.has_kingside_castling_rights(chess.BLACK): castling_value += 4
    if board.has_queenside_castling_rights(chess.BLACK): castling_value += 8
    planes[13].fill(castling_value / 15.0)  # Normalize to [0, 1]
    
    return planes

# ---------- Improved DQN with Convolutional Layers ----------
class ChessCNN(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        # Convolutional layers for spatial understanding
        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_dim)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Input shape: (batch_size, 14, 8, 8)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# ---------- Enhanced Environment Wrapper ----------
class EnhancedChessEnv:
    def __init__(self, base_env):
        self.base_env = base_env
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(14, 8, 8))
        self.action_space = base_env.action_space
        
    def reset(self):
        self.base_env.reset()
        return board_to_planes(self.base_env._board)
    
    def step(self, action):
        next_state, reward, done, info = self.base_env.step(action)
        return board_to_planes(self.base_env._board), reward, done, info
    
    @property
    def _board(self):
        return self.base_env._board

# ---------- Improved Agent ----------
class ImprovedChessDQNAgent:
    def __init__(self,
                 svm_model_path="chess_svm_model.pkl",
                 stockfish_path="stockfish"):

        # --- environment & helper SVM ---
        base_env = SimpleChessEnv(stockfish_path=stockfish_path)
        self.env = EnhancedChessEnv(base_env)
        self.svm_helper = ChessMoveEvaluator(stockfish_path=stockfish_path,
                                             use_gpu=True)
        self.svm_helper.load_model(svm_model_path)

        # --- networks ---
        act_dim = self.env.action_space.n
        print(f"Action space: {act_dim}")

        self.qnet = ChessCNN(act_dim).to(DEVICE)
        self.tgt = ChessCNN(act_dim).to(DEVICE)
        self.tgt.load_state_dict(self.qnet.state_dict())

        self.opt = torch.optim.Adam(self.qnet.parameters(), lr=3e-4)
        self.gamma = 0.995

        # Prioritized experience replay
        self.buffer, self.max_buf = [], 200_000
        self.priorities = np.zeros(self.max_buf)
        self.buffer_pos = 0
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Importance sampling weight
        self.beta_increment = 0.001
        
        # Time diagnostics
        self.time_stats = defaultdict(float)
        
        # Metrics tracking
        self.reward_stats = None
        self.episode_time_stats = None
        self.epsilon_stats = None
        self.loss_history = []

    # --------------------------------------------------
    def _legal_action_ids(self, board):
        """Return a Python list of integer action-ids that are legal now."""
        return [SimpleChessEnv.move_to_action(m) for m in board.legal_moves]
    
    @torch.no_grad()
    def _best_q_legal(self, q_values, legal_ids):
        """
        q_values : 1-D torch tensor length 4096
        legal_ids: list[int]
        returns   : int  (action id with highest Q among legal moves)
        """
        # Create a big negative mask (half-precision safe)
        masked = torch.full_like(q_values, -1e9)
        masked[legal_ids] = q_values[legal_ids]
        return int(torch.argmax(masked))
    # --------------------------------------------------

    # -------- util --------
    def _select_action(self, state_tensor, epsilon: float):
        board = self.env._board  # current board

        # Exploration branch ───
        if np.random.rand() < epsilon:
            # use SVM helper, but fall back to random legal if SVM illegal
            start_time = time.time()
            move, _ = self.svm_helper.find_best_move(board)
            self.time_stats['svm_time'] += time.time() - start_time

            if move not in board.legal_moves:
                print('svm illegal')
                move = random.choice(list(board.legal_moves))
            return SimpleChessEnv.move_to_action(move)

        # Exploitation branch ───
        with torch.no_grad():
            state_t = torch.tensor(state_tensor, dtype=torch.float32,
                                device=DEVICE).unsqueeze(0)
            q_vals = self.qnet(state_t).squeeze(0)

        legal_ids = self._legal_action_ids(board)
        return self._best_q_legal(q_vals, legal_ids)

    def _store(self, transition, priority=None):
        if priority is None:
            priority = 1.0  # Default priority
            
        if len(self.buffer) < self.max_buf:
            self.buffer.append(transition)
            self.priorities = np.append(self.priorities, priority)
        else:
            # Replace old entries
            idx = self.buffer_pos % self.max_buf
            self.buffer[idx] = transition
            self.priorities[idx] = priority
            self.buffer_pos += 1

    def _sample_batch(self, bs=128):
        buffer_len = len(self.buffer)
        if buffer_len < bs:
            return None
            
        # Calculate sampling probabilities
        probs = self.priorities[:buffer_len] ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(buffer_len, bs, p=probs)
        
        # Calculate importance sampling weights
        weights = (buffer_len * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Get batch
        batch = [self.buffer[idx] for idx in indices]
        s, a, r, s2, d = zip(*batch)
        
        # Convert to tensors
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
        
        # Create proper 4D tensors for CNN input (batch_size, channels, height, width)
        return (
            torch.tensor(np.array(s), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(a), dtype=torch.int64, device=DEVICE).unsqueeze(1),
            torch.tensor(np.array(r), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(s2), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(d), dtype=torch.bool, device=DEVICE),
            weights_tensor,
            indices
        )

    def _update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = min(abs(error) + 1e-5, 100)  # Add small constant to avoid zero priority

    def _format_time(self, seconds):
        """Format time in hours, minutes, seconds format"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    # -------- training loop --------
    def train(self,
              episodes=10_000,
              batch_size=128,
              epsilon_start=1.0,
              epsilon_final=0.05,
              reward_shaping=True):
        eps = epsilon_start
        eps_decay = (epsilon_final/epsilon_start) ** (1/episodes)
        update_target_every = 500
        step_count = 0

        global_start_time = time.time()
        episode_times = []  # Store time taken for each episode

        # Store training data
        self.reward_stats = np.empty((episodes,))
        self.episode_time_stats = np.empty((episodes, 5))
        self.epsilon_stats = np.empty((episodes,))
        
        for ep in range(episodes):
            # Reset time stats for this episode
            self.time_stats = defaultdict(float)
            ep_start_time = time.time()
            
            # Reset environment time tracking
            start_time = time.time()
            state, done, ep_reward = self.env.reset(), False, 0
            self.time_stats['env_reset_time'] = time.time() - start_time
            
            step_count_ep = 0  # Count steps in this episode
            last_material_score = self._calculate_material_score(self.env._board)
            
            while not done:
                # Action selection time
                start_time = time.time()
                action = self._select_action(state, eps)
                self.time_stats['action_selection_time'] += time.time() - start_time
                
                # Environment step time
                start_time = time.time()
                next_state, reward, done, info = self.env.step(action)
                self.time_stats['env_step_time'] += time.time() - start_time
                
                # Apply reward shaping if enabled
                if reward_shaping:
                    current_material_score = self._calculate_material_score(self.env._board)
                    material_reward = current_material_score - last_material_score
                    last_material_score = current_material_score
                    
                    # Add positional reward based on piece centrality
                    position_reward = self._calculate_position_reward(self.env._board)
                    
                    # Combine rewards
                    shaped_reward = reward + 0.2 * material_reward + 0.1 * position_reward
                    reward = shaped_reward
                
                # Calculate TD error for prioritized replay (simple estimate)
                state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                next_state_t = torch.tensor(next_state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                
                with torch.no_grad():
                    q_val = self.qnet(state_t)[0, action]
                    next_q = self.tgt(next_state_t).max()
                    target = reward + self.gamma * next_q * (not done)
                    td_error = abs(q_val - target).item()
                
                # Buffer operations time
                start_time = time.time()
                self._store((state, action, reward, next_state, done), priority=td_error)
                state = next_state
                ep_reward += reward
                step_count += 1
                step_count_ep += 1
                self.time_stats['buffer_ops_time'] += time.time() - start_time

                # Learning time
                if len(self.buffer) >= batch_size:
                    start_time = time.time()
                    batch = self._sample_batch(batch_size)
                    
                    if batch is not None:
                        s, a, r, s2, d, weights, indices = batch
                        q_val = self.qnet(s).gather(1, a).squeeze()
                        
                        with torch.no_grad():
                            # Double DQN
                            next_actions = self.qnet(s2).max(1)[1].unsqueeze(1)
                            next_q = self.tgt(s2).gather(1, next_actions).squeeze()
                            tgt_val = r + self.gamma * next_q * (~d)
                        
                        # TD errors for updating priorities
                        td_errors = (q_val - tgt_val).detach().cpu().numpy()
                        
                        # Weighted MSE loss for prioritized replay
                        losses = F.mse_loss(q_val, tgt_val, reduction='none')
                        loss = (weights * losses).mean()
                        
                        self.opt.zero_grad()
                        loss.backward()
                        # Gradient clipping
                        nn.utils.clip_grad_norm_(self.qnet.parameters(), 1.0)
                        self.opt.step()
                        
                        # Update priorities
                        self._update_priorities(indices, td_errors)
                        
                        # Update beta for importance sampling
                        self.beta = min(1.0, self.beta + self.beta_increment)
                        
                        # Save loss for tracking
                        self.loss_history.append(loss.item())
                        
                    self.time_stats['learning_time'] += time.time() - start_time

                    # Target network update
                    if step_count % update_target_every == 0:
                        start_time = time.time()
                        self.tgt.load_state_dict(self.qnet.state_dict())
                        self.time_stats['target_update_time'] += time.time() - start_time

            # Record episode time
            ep_time = time.time() - ep_start_time
            episode_times.append(ep_time)
            
            # Calculate average episode time and ETA
            avg_ep_time = sum(episode_times[-100:]) / min(len(episode_times), 100)  # Moving average of last 100 episodes
            remaining_eps = episodes - (ep + 1)
            eta_seconds = avg_ep_time * remaining_eps
            
            # Calculate percentage breakdown
            total_time = sum(self.time_stats.values())
            if total_time > 0:  # Avoid division by zero
                percentages = {k: (v / total_time) * 100 for k, v in self.time_stats.items()}
            else:
                percentages = {k: 0 for k in self.time_stats}

            # Schedule epsilon
            eps = max(epsilon_final, eps * eps_decay)

            # Data collection
            self.reward_stats[ep] = ep_reward
            self.episode_time_stats[ep] = [
                ep_time,  # ← raw float, e.g., 95.31 seconds
                self.time_stats['env_step_time'],
                self.time_stats['action_selection_time'],
                self.time_stats.get('learning_time', 0),
                self.time_stats['buffer_ops_time']
            ]
            self.epsilon_stats[ep] = eps

            # Comprehensive logging every N episodes
            log_frequency = 10  # Adjust as needed
            if (ep+1) % log_frequency == 0:
                elapsed = time.time() - global_start_time
                progress_percent = ((ep + 1) / episodes) * 100
                
                # Calculate average loss over recent episodes
                avg_loss = np.mean(self.loss_history[-1000:]) if self.loss_history else 0
                
                print(f"\n{'='*60}")
                print(f"Episode {ep+1:>5}/{episodes} ({progress_percent:.1f}% complete)")
                print(f"{'='*60}")
                print(f"ε={eps:.4f}  Reward={ep_reward:.3f}  Steps in episode={step_count_ep}  Avg Loss={avg_loss:.5f}")
                print(f"Episode time: {self._format_time(ep_time)} ({ep_time:.2f}s)")
                print(f"Elapsed time: {self._format_time(elapsed)}")
                print(f"ETA: {self._format_time(eta_seconds)}")
                print(f"\nPerformance breakdown:")
                if 'svm_time' in self.time_stats:
                    print(f"  SVM time: {self.time_stats['svm_time']:.2f}s ({percentages['svm_time']:.1f}%)")
                print(f"  Environment step time: {self.time_stats['env_step_time']:.2f}s ({percentages['env_step_time']:.1f}%)")
                print(f"  Action selection time: {self.time_stats['action_selection_time']:.2f}s ({percentages['action_selection_time']:.1f}%)")
                print(f"  Learning time: {self.time_stats.get('learning_time', 0):.2f}s ({percentages.get('learning_time', 0):.1f}%)")
                print(f"  Buffer operations: {self.time_stats['buffer_ops_time']:.2f}s ({percentages['buffer_ops_time']:.1f}%)")
                if 'target_update_time' in self.time_stats:
                    print(f"  Target network updates: {self.time_stats['target_update_time']:.2f}s ({percentages['target_update_time']:.1f}%)")
                print(f"{'='*60}\n")
                
                # Quick evaluation of policy
                if (ep+1) % (log_frequency * 5) == 0:
                    self._evaluate_policy()

    def _calculate_material_score(self, board):
        """Calculate material score for the current board state."""
        piece_values = {'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0,
                        'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0}
        
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                score += piece_values[piece.symbol()]
                
        return score
    
    def _calculate_position_reward(self, board):
        """Calculate reward based on piece positioning."""
        # Center control bonus
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        center_control = 0
        
        # Count pieces controlling or occupying center
        for square in center_squares:
            piece = board.piece_at(square)
            # Bonus for occupying center
            if piece:
                color_mult = 1 if piece.color == chess.WHITE else -1
                center_control += 0.1 * color_mult
            
            # Bonus for attacking center
            attackers = board.attackers(chess.WHITE, square)
            center_control += 0.05 * len(attackers)
            
            attackers = board.attackers(chess.BLACK, square)
            center_control -= 0.05 * len(attackers)
        
        # Pawn structure bonus
        pawn_structure = 0
        # Simple implementation - just count doubled/isolated pawns
        for file in range(8):
            white_pawns = 0
            black_pawns = 0
            for rank in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN:
                    if piece.color == chess.WHITE:
                        white_pawns += 1
                    else:
                        black_pawns += 1
            
            # Penalty for doubled pawns
            if white_pawns > 1:
                pawn_structure -= 0.1 * (white_pawns - 1)
            if black_pawns > 1:
                pawn_structure += 0.1 * (black_pawns - 1)
                
        # Return combined position score
        return center_control + pawn_structure
    
    def _evaluate_policy(self, games=5):
        """Evaluate the current policy by playing a few games."""
        print("\nEvaluating current policy...")
        total_reward = 0
        total_steps = 0
        
        for game in range(games):
            state = self.env.reset()
            done = False
            game_reward = 0
            steps = 0
            
            while not done and steps < 100:  # Cap at 100 moves to prevent infinite games
                action = self._select_action(state, epsilon=0.05)  # Small epsilon for some exploration
                state, reward, done, _ = self.env.step(action)
                game_reward += reward
                steps += 1
                
            total_reward += game_reward
            total_steps += steps
            
        avg_reward = total_reward / games
        avg_steps = total_steps / games
        print(f"Evaluation complete: Avg Reward={avg_reward:.3f}, Avg Steps={avg_steps:.1f}")
        return avg_reward

    # -------- persistence --------
    def save(self, path="improved_dqn_chess.pth", data_path="improved_dqn_training_stats.csv"):
        torch.save({
            "state_dict": self.qnet.state_dict(),
            "act_dim": self.env.action_space.n
        }, path)
        
        # Save data from training
        df = pd.DataFrame({
            "episode": np.arange(len(self.reward_stats)),
            "reward": self.reward_stats,
            "epsilon": self.epsilon_stats,
            "episode_time": self.episode_time_stats[:, 0],
            "env_step_time": self.episode_time_stats[:, 1],
            "action_selection_time": self.episode_time_stats[:, 2],
            "learning_time": self.episode_time_stats[:, 3],
            "buffer_ops_time": self.episode_time_stats[:, 4],
        })
        df.to_csv(data_path, index=False)
        print(f"Data saved to {data_path}")
        
        # Save loss history
        np.save(path.replace('.pth', '_loss_history.npy'), np.array(self.loss_history))
        print(f"Loss history saved to {path.replace('.pth', '_loss_history.npy')}")

    def load(self, path="improved_dqn_chess.pth"):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.qnet.load_state_dict(checkpoint["state_dict"])
        self.tgt.load_state_dict(checkpoint["state_dict"])
        print(f"Model loaded from {path}")
        return self


# -------------------------------------------------
if __name__ == "__main__":
    trainer = ImprovedChessDQNAgent(svm_model_path="chess_svm_model_5000_games_featuresV3_architectureV2_GPUtrained.pkl",
                                   stockfish_path="/usr/games/stockfish")
    trainer.train(episodes=5000, reward_shaping=True)
    trainer.save(path="Improved_DQN_agent_5000episodes.pth", data_path="improved_dqn_training_stats_5000ep.csv")
    exit(0)