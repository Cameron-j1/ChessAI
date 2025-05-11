"""
Gym-style environment that lets an RL agent play chess.
Reward = –|Stockfish_best – agent_move|  (pawns)
+1 bonus if the agent actually picks the Stockfish best move.

Action space  : 4 096 discrete actions (from-square × to-square)
Observation   : 779-dim feature vector re-using the feature extractor
                already implemented in chess_svm_model.py
"""

import gym
import numpy as np
import chess
import chess.engine
from chess_svm_model import ChessMoveEvaluator      # re-use its feature code

_STOCKFISH = "stockfish"      # adjust if needed
_EVAL_TIME  = 0.001            # seconds per Stockfish probe
_MAX_PLIES  = 200             # hard cutoff to avoid endless games
_ACTIONS    = 64 * 64         # 4 096

class SimpleChessEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    # ---------- utility: (move) ↔ (int) ----------
    @staticmethod
    def move_to_action(move: chess.Move) -> int:
        """Encode a move (ignoring promotions) into 0…4 095."""
        return move.from_square * 64 + move.to_square

    @staticmethod
    def action_to_move(idx: int, board: chess.Board) -> chess.Move:
        """Decode int → chess.Move, auto-promoting pawns to queen."""
        frm, to = divmod(idx, 64)
        if board.piece_at(frm) and board.piece_at(frm).piece_type == chess.PAWN \
           and chess.square_rank(to) in (0, 7):
            return chess.Move(frm, to, promotion=chess.QUEEN)
        return chess.Move(frm, to)
    # ---------------------------------------------

    def __init__(self,
                 stockfish_path: str = _STOCKFISH,
                 eval_time: float = _EVAL_TIME,
                 max_plies: int = _MAX_PLIES):
        super().__init__()
        self.action_space      = gym.spaces.Discrete(_ACTIONS)
        self.observation_space = gym.spaces.Box(low=-20, high=20,
                                                shape=(779,), dtype=np.float32)

        self._engine   = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self._evaluator = ChessMoveEvaluator(stockfish_path=stockfish_path,
                                             use_gpu=False)   # features only
        self._eval_time = eval_time
        self._max_plies  = max_plies
        self.reset()

    # ---------- core helpers ----------
    def _features(self) -> np.ndarray:
        return np.asarray(
            self._evaluator._extract_board_features(self._board), dtype=np.float32
        )

    def _sf_eval(self, board: chess.Board) -> float:
        """Return Stockfish centipawn eval → pawns (positive = side to move better)."""
        info = self._engine.analyse(board, chess.engine.Limit(time=self._eval_time))
        return info["score"].relative.score(mate_score=10000) / 100.0

    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._board = chess.Board()
        self._plies  = 0
        return self._features()

    def step(self, action: int):
        move = self.action_to_move(action, self._board)

        # illegal move → instant failure
        if move not in self._board.legal_moves:
            obs   = self._features()
            reward = -5.0
            done   = True
            return obs, reward, done, {}

        # Stockfish data BEFORE we play
        best_info = self._engine.analyse(self._board,
                                         chess.engine.Limit(time=self._eval_time))
        best_move = best_info["pv"][0]
        eval_best_board = None  # compute later

        # Play the agent's move
        board_before = self._board.copy()
        self._board.push(move)
        self._plies += 1

        # Evaluation AFTER agent move
        eval_after_agent = self._sf_eval(self._board)

        # Evaluation AFTER best move (need separate board)
        board_best = board_before.copy()
        board_best.push(best_move)
        eval_after_best = self._sf_eval(board_best)

        # reward: closer to best is better; +1 bonus if identical
        delta   = abs(eval_after_best - eval_after_agent)
        reward  = -delta
        if move == best_move:
            reward += 1.0

        done = self._board.is_game_over(claim_draw=True) or self._plies >= self._max_plies
        obs  = self._features()
        return obs, reward, done, {}

    def render(self, mode="human"):
        if mode == "human":
            print(self._board)

    def close(self):
        self._engine.quit()
