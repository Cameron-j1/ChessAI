"""
Evaluate a trained DQN-based chess agent.

Two test modes
--------------
1. vs Stockfish ― play N full games against a configurable-Elo engine
   *  win / draw / loss rate
   *  % of agent moves that exactly match Stockfish's own best move
2. vs Random ― play N games against an opponent that picks uniformly
   among legal moves
   *  win / draw / loss rate

The DQN is the same architecture used in train_agent.py.
"""

import random, math, time, argparse, pathlib
import numpy as np
import torch
import chess
import chess.engine

from simple_chess_env import SimpleChessEnv   # for move/action helpers
from train_DQN_agent      import DQN              # same network definition
from chess_svm_model  import ChessMoveEvaluator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SF_BIN = "stockfish"          # modify if needed
SF_TIME = 0.001                # seconds per Stockfish move
MAX_PLIES = 200               # safety cut-off

# --------------------------------------------------------------------------- #
#  Utilities                                                                  #
# --------------------------------------------------------------------------- #

def load_model(ckpt_path: str) -> DQN:
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    net  = DQN(ckpt["obs_dim"], ckpt["act_dim"]).to(DEVICE)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net

def board_to_state(board: chess.Board,
                   evaluator: ChessMoveEvaluator) -> np.ndarray:
    """779-D feature vector (same as used during training)."""
    return np.asarray(
        evaluator._extract_board_features(board), dtype=np.float32
    )

def action_to_move(action: int, board: chess.Board) -> chess.Move:
    return SimpleChessEnv.action_to_move(action, board)

def move_to_action(move: chess.Move) -> int:
    return SimpleChessEnv.move_to_action(move)

# --------------------------------------------------------------------------- #
#  Tester                                                                     #
# --------------------------------------------------------------------------- #

class ChessDQNTester:
    def __init__(self,
                 model_path   = "dqn_chess.pth",
                 stockfish_bin= SF_BIN,
                 sf_time      = SF_TIME):

        self.net  = load_model(model_path)
        self.evaluator = ChessMoveEvaluator(stockfish_path=stockfish_bin,
                                            use_gpu=False)
        self.stockfish_bin = stockfish_bin
        self.sf_time       = sf_time

    # --------------- low-level helpers ----------------
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
    
    @torch.no_grad()
    def _choose_agent_move(self, board):
        feat   = board_to_state(board, self.evaluator)
        state  = torch.tensor(feat, dtype=torch.float32,
                            device=DEVICE).unsqueeze(0)
        q_vals = self.net(state).squeeze(0)

        legal_ids = [move_to_action(m) for m in board.legal_moves]
        best_id   = self._best_q_legal(q_vals, legal_ids)   # reuse helper
        return action_to_move(best_id, board)

    def _best_move_by_sf(self, engine, board) -> chess.Move:
        info = engine.analyse(board, chess.engine.Limit(time=self.sf_time))
        return info["pv"][0]

    # --------------- test v Stockfish -----------------
    def play_vs_stockfish(self,
                          games     = 50,
                          sf_elo    = 1500,
                          verbose   = False):
        wins = draws = losses = 0
        total_moves = match_moves = 0

        with chess.engine.SimpleEngine.popen_uci(self.stockfish_bin) as eng:
            eng.configure({"UCI_LimitStrength": True,
                           "UCI_Elo": sf_elo})

            for g in range(1, games+1):
                board, plies = chess.Board(), 0
                if verbose:
                    print(f"\n=== Game {g} ===")
                
                game_result = None  # Track if we got a result for this game
                
                while not board.is_game_over(claim_draw=True) and plies < MAX_PLIES:
                    # ----- Agent (White if turn==WHITE else Black) -----
                    agent_move = self._choose_agent_move(board)
                    if agent_move not in board.legal_moves:
                        # illegal ⇒ immediate loss
                        if verbose:
                            print("Agent made illegal move.")
                        losses += 1
                        game_result = "illegal"
                        break

                    sf_best = self._best_move_by_sf(eng, board)
                    if agent_move == sf_best:
                        match_moves += 1
                    total_moves += 1

                    board.push(agent_move)
                    plies += 1
                    if board.is_game_over(): break

                    # ----- Stockfish reply -----
                    sf_reply = eng.play(board, chess.engine.Limit(time=self.sf_time)).move
                    board.push(sf_reply)
                    plies += 1

                    if verbose:
                        print(board.peek())  # last move

                # outcome
                if game_result != "illegal":  # Only evaluate if not already marked as loss due to illegal move
                    if board.is_game_over():
                        result = board.result()  # "1-0", "0-1", "1/2-1/2"
                        # --- decide result from P-G-N string ---------------------------------
                        if result == "1-0":          # White wins
                            wins += 1
                        elif result == "0-1":        # Black wins
                            losses += 1
                        else:                        # "1/2-1/2"
                            draws += 1
                        # ---------------------------------------------------------------------
                    elif plies >= MAX_PLIES:
                        draws += 1  # treat cutoff as draw

        print(f"\n=== vs Stockfish (Elo {sf_elo}) ===")
        self._print_summary(wins, draws, losses, total_moves, match_moves)

    # --------------- test v Random --------------------
    def play_vs_random(self,
                       games   = 50,
                       verbose = False):
        wins = draws = losses = 0
        skipped_games = []  # Track any games that might be skipped

        for g in range(1, games+1):
            board, plies = chess.Board(), 0
            if verbose:
                print(f"\n=== Game {g} (Random) ===")
            
            game_counted = False  # Flag to track if we've counted this game
            
            try:
                while not board.is_game_over(claim_draw=True) and plies < MAX_PLIES:
                    # Agent move
                    move = self._choose_agent_move(board)
                    if move not in board.legal_moves:
                        losses += 1
                        game_counted = True
                        print("loss (illegal move)")
                        break
                    board.push(move)
                    plies += 1
                    if board.is_game_over(): break

                    # Random reply
                    legal_moves = list(board.legal_moves)
                    if legal_moves:  # Check if there are legal moves
                        rand_move = random.choice(legal_moves)
                        board.push(rand_move)
                        plies += 1
                    else:
                        # No legal moves for random player (checkmate or stalemate)
                        break

                    if verbose:
                        print(board.peek())

                # Only evaluate the outcome if we haven't already counted this game
                if not game_counted:
                    if board.is_game_over():
                        result = board.result()
                        # --- decide result from P-G-N string ---------------------------------
                        if result == "1-0":          # White wins
                            wins += 1
                        elif result == "0-1":        # Black wins
                            losses += 1
                        else:                        # "1/2-1/2"
                            draws += 1
                        
                        # ---------------------------------------------------------------------
                    elif plies >= MAX_PLIES:
                        draws += 1
            except Exception as e:
                # Catch any unexpected errors and count the game as a draw
                draws += 1
                print(f"draw (error: {str(e)})")
                skipped_games.append(g)

            # Double-check that the game was counted
            if not game_counted and not board.is_game_over() and plies < MAX_PLIES:
                draws += 1
                skipped_games.append(g)

        print("\n=== vs Random ===")
        self._print_summary(wins, draws, losses)

    # --------------- util -----------------------------
    @staticmethod
    def _print_summary(w, d, l, total_moves=None, match_moves=None):
        games = w + d + l
        print(f"Games: {games} | W {w} / D {d} / L {l}")
        if games > 0:  # Avoid division by zero
            print(f"Win rate: {w/games*100:.1f}% | Draw rate: {d/games*100:.1f}% | Loss rate: {l/games*100:.1f}%")
        if total_moves:
            acc = 100.0 * match_moves / total_moves if total_moves else 0.0
            print(f"Move-match vs Stockfish: {match_moves}/{total_moves} "
                  f"({acc:.2f} %)")
        print("-" * 40)

# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="dqn_chess.pth",
                        help="path to saved DQN checkpoint")
    parser.add_argument("--games",  type=int, default=50,
                        help="games per test")
    parser.add_argument("--elo",    type=int, default=1500,
                        help="Stockfish Elo for vs-engine test")
    parser.add_argument("--verbose", action="store_true",
                        help="print every move")
    parser.add_argument("--mode", choices=["stockfish", "random", "both"], 
                        default="random", help="test mode")
    args = parser.parse_args()

    print(f"Testing {args.model} for {args.games} games")
    
    try:
        tester = ChessDQNTester(model_path=args.model)
        
        if args.mode == "stockfish" or args.mode == "both":
            print(f"\nTesting against Stockfish (Elo: {args.elo})...")
            tester.play_vs_stockfish(games=args.games, sf_elo=args.elo,
                                verbose=args.verbose)
        if args.mode == "random" or args.mode == "both":
            print(f"\nTesting against Random player...")
            tester.play_vs_random(games=args.games, verbose=args.verbose)
            
        print(f"\nTesting completed successfully.")
    except Exception as e:
        print(f"\nError during testing: {str(e)}")

# python test_DQN_agent.py --model DQN_agent_10episodes.pth --games 100 --elo 1350 --mode "both"

