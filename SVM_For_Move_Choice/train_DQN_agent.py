"""
DQN trainer for the SimpleChessEnv.
ε-greedy schedule starts by delegating to a pre-trained ChessMoveEvaluator,
then gradually shifts to the neural-net policy.
Enhanced with detailed time diagnostics.
"""

import random, time, math, pickle, gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict

from simple_chess_env import SimpleChessEnv
from chess_svm_model import ChessMoveEvaluator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- DQN ----------
class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 512),  nn.ReLU(),
            nn.Linear(512,  output_dim),
        )
    def forward(self, x):  return self.net(x)

# ---------- Agent ----------
class ChessDQNAgent:
    def __init__(self,
                 svm_model_path="chess_svm_model.pkl",
                 stockfish_path="stockfish"):

        # --- environment & helper SVM ---
        self.env         = SimpleChessEnv(stockfish_path=stockfish_path)
        self.svm_helper  = ChessMoveEvaluator(stockfish_path=stockfish_path,
                                              use_gpu=True)
        self.svm_helper.load_model(svm_model_path)

        # --- networks ---
        obs_dim   = self.env.observation_space.shape[0]
        act_dim   = self.env.action_space.n
        self.qnet = DQN(obs_dim, act_dim).to(DEVICE)
        self.tgt  = DQN(obs_dim, act_dim).to(DEVICE)
        self.tgt.load_state_dict(self.qnet.state_dict())

        self.opt   = torch.optim.Adam(self.qnet.parameters(), lr=1e-4)
        self.gamma = 0.995

        # experience replay
        self.buffer, self.max_buf = [], 150_000
        
        # Time diagnostics
        self.time_stats = defaultdict(float)

        # train diagnostics


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
        # Create a big negative mask   (half-precision safe)
        masked = torch.full_like(q_values, -1e9)
        masked[legal_ids] = q_values[legal_ids]
        return int(torch.argmax(masked))
    # --------------------------------------------------

    # -------- util --------
    def _select_action(self, state_np, epsilon: float):
        board = self.env._board                    # current board

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
        state_t = torch.tensor(state_np, dtype=torch.float32,
                            device=DEVICE).unsqueeze(0)
        q_vals   = self.qnet(state_t).squeeze(0)

        legal_ids = self._legal_action_ids(board)
        return self._best_q_legal(q_vals, legal_ids)

    def _store(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) > self.max_buf:
            self.buffer.pop(0)

    def _sample_batch(self, bs=128):
        batch = random.sample(self.buffer, bs)
        s, a, r, s2, d = zip(*batch)
        
        # Convert lists to numpy arrays first, then to tensors (as recommended)
        return (torch.tensor(np.array(s),  dtype=torch.float32, device=DEVICE),
                torch.tensor(np.array(a),  dtype=torch.int64,   device=DEVICE).unsqueeze(1),
                torch.tensor(np.array(r),  dtype=torch.float32, device=DEVICE),
                torch.tensor(np.array(s2), dtype=torch.float32, device=DEVICE),
                torch.tensor(np.array(d),  dtype=torch.bool,    device=DEVICE))

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
              epsilon_final=0.05):
        eps = epsilon_start
        eps_decay = (epsilon_final/epsilon_start) ** (1/episodes)
        update_target_every = 1000
        step_count = 0

        global_start_time = time.time()
        episode_times = []  # Store time taken for each episode

        #store training data
        self.reward_stats = np.empty((episodes,))
        self.episode_time_stats = np.empty((episodes, 5))
        self.epsilon_stats = np.empty((episodes,))
        
        for ep in range(episodes):
            # Reset time stats for this episode
            self.time_stats = defaultdict(float)
            ep_start_time = time.time()
            
            # print(f"Starting episode: {ep}")
            
            # Reset environment time tracking
            start_time = time.time()
            state, done, ep_reward = self.env.reset(), False, 0
            self.time_stats['env_reset_time'] = time.time() - start_time
            
            step_count_ep = 0  # Count steps in this episode
            
            while not done:
                # Action selection time
                start_time = time.time()
                action = self._select_action(state, eps)
                self.time_stats['action_selection_time'] += time.time() - start_time
                
                # Environment step time
                start_time = time.time()
                next_state, reward, done, _ = self.env.step(action)
                self.time_stats['env_step_time'] += time.time() - start_time
                
                # Buffer operations time
                start_time = time.time()
                self._store((state, action, reward, next_state, done))
                state = next_state
                ep_reward += reward
                step_count += 1
                step_count_ep += 1
                self.time_stats['buffer_ops_time'] += time.time() - start_time

                # Learning time
                if len(self.buffer) >= batch_size:
                    start_time = time.time()
                    s, a, r, s2, d = self._sample_batch(batch_size)
                    q_val   = self.qnet(s).gather(1, a).squeeze()
                    with torch.no_grad():
                        tgt_max = self.tgt(s2).max(1)[0]
                        tgt_val = r + self.gamma * tgt_max * (~d)
                    loss = nn.functional.mse_loss(q_val, tgt_val)

                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.qnet.parameters(), 1.0)
                    self.opt.step()
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
            avg_ep_time = sum(episode_times) / len(episode_times)
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

            #data collection
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
            log_frequency = 1  # Set to 1 to log every episode or higher for less frequent logging
            if (ep+1) % log_frequency == 0:
                elapsed = time.time() - global_start_time
                progress_percent = ((ep + 1) / episodes) * 100
                
                print(f"\n{'='*60}")
                print(f"Episode {ep+1:>5}/{episodes} ({progress_percent:.1f}% complete)")
                print(f"{'='*60}")
                print(f"ε={eps:.4f}  Reward={ep_reward:.3f}  Steps in episode={step_count_ep}")
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

    # -------- persistence --------
    def save(self, path="dqn_chess.pth", data_path="dqn_training_stats.csv"):
        torch.save({"state_dict": self.qnet.state_dict(),
                    "obs_dim": self.env.observation_space.shape[0],
                    "act_dim": self.env.action_space.n}, path)
        
        #save data from training
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
        print(f"data saved to {data_path}")


# -------------------------------------------------
if __name__ == "__main__":
    trainer = ChessDQNAgent(svm_model_path="chess_svm_model_5000_games_featuresV3_architectureV2_GPUtrained.pkl",
                            stockfish_path="/usr/games/stockfish")
    trainer.train(episodes=1000)
    trainer.save(path="DQN_agent_1000episodes.pth", data_path="dqn_training_stats_1000ep.csv")
    exit(0)