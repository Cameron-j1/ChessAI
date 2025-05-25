import chess
import chess.pgn
import chess.engine
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import io
import joblib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# GPU imports - CuML implementation
import cudf
import cuml
from cuml.svm import SVR as cuSVR
from cuml.preprocessing import StandardScaler as cuStandardScaler

from sklearn.pipeline import Pipeline  # not from cuml

class ChessPositionEvaluator:
    def __init__(self, stockfish_path="stockfish", use_gpu=True):
        """
        Initialize the Chess Position Evaluator
        
        Args:
            stockfish_path (str): Path to Stockfish engine executable
            use_gpu (bool): Whether to use GPU acceleration
        """
        self.stockfish_path = stockfish_path
        self.model = None
        self.scaler = None
        self.pst = self._generate_simple_pst()
        self.use_gpu = use_gpu
        
        # Print GPU info if using GPU
        if self.use_gpu:
            try:
                import cupy as cp
                print(f"GPU detected: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
                print(f"CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
            except ImportError:
                print("CuPy not found. Install with: pip install cupy-cuda11x (replace x with your CUDA version)")
            except Exception as e:
                print(f"Error detecting GPU: {e}")
                self.use_gpu = False
                print("Falling back to CPU mode")

    def _generate_simple_pst(self):
        """
        Create a very light-weight piece-square-table (PST).
        """
        # Distance of every square (0…63) from board centre
        centre_files = np.array([3.5, 4.5])        # d / e files
        centre_ranks = np.array([3.5, 4.5])        # 4th / 5th ranks
        dist = np.zeros(64, dtype=np.float32)
    
        for sq in range(64):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            dist[sq] = np.min(np.abs(f - centre_files) + np.abs(r - centre_ranks))
    
        # Normalise to [-1, +1] (0 = very central, 4 = corner)
        dist = 1.0 - (dist / 4.0) * 2.0        # 0 → +1   |   4 → -1
    
        # Multiply by piece values
        pst = {}
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        for ptype, pval in piece_values.items():
            pst[ptype] = dist * pval
    
        return pst
        
    def _get_piece_value(self, piece_type):
        """Return the material value of a piece"""
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King's value isn't used in material counting
        }
        return values.get(piece_type, 0)
    
    def _square_to_feature(self, square):
        # Converts square index (0–63) to a number between 0 and 63; returns -1 if None
        return square if square is not None else -1

    def _pst_scores(self, board: chess.Board, pst_tables) -> tuple[float, float]:
        """
        Sum of PST values for each side (white, black).
        """
        white_score = 0.0
        black_score = 0.0
        for sq, piece in board.piece_map().items():
            val = pst_tables[piece.piece_type][sq]
            if piece.color == chess.WHITE:
                white_score += val
            else:
                # mirror for black
                mirror_sq = chess.square_mirror(sq)
                black_score += pst_tables[piece.piece_type][mirror_sq]
        return white_score, black_score
    
    def _manhattan(self, a: int, b: int) -> int:
        """Manhattan distance between two squares (0…63)."""
        return abs(chess.square_file(a) - chess.square_file(b)) + \
               abs(chess.square_rank(a) - chess.square_rank(b))
    
    def _material_on_board(self, board: chess.Board) -> float:
        """Calculate material balance (positive favors white, negative favors black)"""
        vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9}
        total = 0
        for piece in board.piece_map().values():
            total += vals.get(piece.piece_type, 0) * (1 if piece.color else -1)
        return total

    def _extract_position_features(self, board):
        """
        Extract comprehensive features from a chess board position
        
        Args:
            board (chess.Board): Current board position
            
        Returns:
            list: Features representing the board state
        """
        features = []
        
        # Material count and advantage
        white_material = 0
        black_material = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                if piece.color == chess.WHITE:
                    white_material += self._get_piece_value(piece.piece_type)
                else:
                    black_material += self._get_piece_value(piece.piece_type)
        
        features.append(white_material - black_material)  # Material advantage
        features.append(white_material + black_material)  # Total material (game phase indicator)
        
        # Piece-square table scores
        w_pst, b_pst = self._pst_scores(board, self.pst)
        features.append(w_pst - b_pst)  # Positional advantage
        
        # Control of the center (e4, d4, e5, d5)
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        white_center_control = sum(1 for sq in center_squares if board.is_attacked_by(chess.WHITE, sq))
        black_center_control = sum(1 for sq in center_squares if board.is_attacked_by(chess.BLACK, sq))
        features.append(white_center_control - black_center_control)
        
        # Mobility (number of legal moves)
        original_turn = board.turn
        
        board.turn = chess.WHITE
        white_mobility = len(list(board.legal_moves))
        
        board.turn = chess.BLACK
        black_mobility = len(list(board.legal_moves))
        
        # Restore original turn
        board.turn = original_turn
        
        features.append(white_mobility - black_mobility)  # Mobility advantage
        features.append(white_mobility + black_mobility)  # Total mobility
            
        # King safety (attacks near king)
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        
        if white_king_square:
            white_king_neighborhood = chess.SquareSet(chess.BB_KING_ATTACKS[white_king_square])
            white_king_attacks = sum(1 for sq in white_king_neighborhood if board.is_attacked_by(chess.BLACK, sq))
        else:
            white_king_attacks = 0
            
        if black_king_square:
            black_king_neighborhood = chess.SquareSet(chess.BB_KING_ATTACKS[black_king_square])
            black_king_attacks = sum(1 for sq in black_king_neighborhood if board.is_attacked_by(chess.WHITE, sq))
        else:
            black_king_attacks = 0
            
        features.append(black_king_attacks)
        features.append(white_king_attacks)  # King safety
                
        # Pawn structure features
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        
        # Doubled pawns
        white_doubled = 0
        black_doubled = 0
        for file_idx in range(8):
            file_mask = chess.BB_FILES[file_idx]
            white_pawns_in_file = len(white_pawns & file_mask)
            black_pawns_in_file = len(black_pawns & file_mask)
            if white_pawns_in_file > 1:
                white_doubled += white_pawns_in_file - 1
            if black_pawns_in_file > 1:
                black_doubled += black_pawns_in_file - 1
        
        features.append(black_doubled - white_doubled)  # Doubled pawns advantage
        
        # Passed pawns
        white_passed = 0
        black_passed = 0
        
        # Convert SquareSet to bitboard (int) for scan_forward()
        white_pawns_bb = white_pawns.mask
        black_pawns_bb = black_pawns.mask
        
        for sq in chess.scan_forward(white_pawns_bb):
            # Check if no black pawns in front or adjacent files
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            is_passed = True
            
            for r in range(rank + 1, 8):
                for f in range(max(0, file - 1), min(8, file + 2)):
                    blocker_sq = chess.square(f, r)
                    if bool(black_pawns & chess.BB_SQUARES[blocker_sq]):
                        is_passed = False
                        break
                if not is_passed:
                    break
                    
            if is_passed:
                white_passed += 1
                
        for sq in chess.scan_forward(black_pawns_bb):
            # Check if no white pawns in front or adjacent files
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            is_passed = True
            
            for r in range(rank - 1, -1, -1):
                for f in range(max(0, file - 1), min(8, file + 2)):
                    blocker_sq = chess.square(f, r)
                    if bool(white_pawns & chess.BB_SQUARES[blocker_sq]):
                        is_passed = False
                        break
                if not is_passed:
                    break
                    
            if is_passed:
                black_passed += 1
                
        features.append(white_passed - black_passed)  # Passed pawns advantage
        
        # Check status
        features.append(1 if board.is_check() else 0)
        
        # Castling rights
        castling_rights = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            castling_rights += 1
        if board.has_queenside_castling_rights(chess.WHITE):
            castling_rights += 1
        if board.has_kingside_castling_rights(chess.BLACK):
            castling_rights -= 1
        if board.has_queenside_castling_rights(chess.BLACK):
            castling_rights -= 1
        features.append(castling_rights)
        
        # Space control (squares controlled beyond 3rd/5th rank)
        white_space = 0
        black_space = 0
        
        for r in range(4, 8):  # Ranks 5-8 (black territory)
            for f in range(8):
                sq = chess.square(f, r)
                if board.is_attacked_by(chess.WHITE, sq):
                    white_space += 1
                    
        for r in range(4):  # Ranks 1-4 (white territory)
            for f in range(8):
                sq = chess.square(f, r)
                if board.is_attacked_by(chess.BLACK, sq):
                    black_space += 1
                    
        features.append(white_space - black_space)  # Space control advantage
        
        # Development (minor pieces moved from starting squares)
        white_development = 0
        black_development = 0
        
        # White knights developed?
        if not board.piece_at(chess.B1) or board.piece_at(chess.B1).piece_type != chess.KNIGHT:
            white_development += 1
        if not board.piece_at(chess.G1) or board.piece_at(chess.G1).piece_type != chess.KNIGHT:
            white_development += 1
            
        # White bishops developed?
        if not board.piece_at(chess.C1) or board.piece_at(chess.C1).piece_type != chess.BISHOP:
            white_development += 1
        if not board.piece_at(chess.F1) or board.piece_at(chess.F1).piece_type != chess.BISHOP:
            white_development += 1
            
        # Black knights developed?
        if not board.piece_at(chess.B8) or board.piece_at(chess.B8).piece_type != chess.KNIGHT:
            black_development += 1
        if not board.piece_at(chess.G8) or board.piece_at(chess.G8).piece_type != chess.KNIGHT:
            black_development += 1
            
        # Black bishops developed?
        if not board.piece_at(chess.C8) or board.piece_at(chess.C8).piece_type != chess.BISHOP:
            black_development += 1
        if not board.piece_at(chess.F8) or board.piece_at(chess.F8).piece_type != chess.BISHOP:
            black_development += 1
            
        features.append(white_development - black_development)  # Development advantage
        
        # Current side to move
        features.append(1 if board.turn == chess.WHITE else -1)
        
        # Halfmove clock (for 50-move rule)
        features.append(board.halfmove_clock / 100.0)  # Normalize to [0, 0.5]
        
        # Fullmove number (game progression)
        features.append(min(80, board.fullmove_number) / 80.0)  # Normalize to [0, 1]
        
        return features
            
    def prepare_training_data(self, pgn_files, num_games=None, max_workers=10):
        """
        Parallelized version of prepare_training_data.
        """
        features_list = []
        evaluations = []
        games_processed = 0

        position_cache = {}

        def process_game(game):
            board = game.board()
            game_features = []
            game_evals = []
            move_count = 0

            # Create a new Stockfish instance per thread
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
                engine.configure({"Threads": 1, "Hash": 64})

                for move in game.mainline_moves():
                    move_count += 1
                    if move_count % 3 != 0:
                        board.push(move)
                        continue

                    features = self._extract_position_features(board)
                    fen = board.fen()

                    if fen in position_cache:
                        eval_score = position_cache[fen]
                    else:
                        eval_score = self._get_stockfish_evaluation(board, engine, time_limit=0.01)
                        position_cache[fen] = eval_score

                    game_features.append(features)
                    game_evals.append(eval_score)

                    board.push(move)
            return game_features, game_evals

        all_games = []
        for pgn_file in pgn_files:
            with open(pgn_file) as pgn:
                while True:
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break
                    all_games.append(game)
                    games_processed += 1
                    if num_games is not None and games_processed >= num_games:
                        break

        print(f"Total games loaded: {len(all_games)}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_game, game) for game in all_games]
            for f in tqdm(as_completed(futures), total=len(futures)):
                ftrs, evls = f.result()
                features_list.extend(ftrs)
                evaluations.extend(evls)

        return np.array(features_list), np.array(evaluations)

    def _get_stockfish_evaluation(self, board, engine=None, time_limit=0.01):
        """
        Get position evaluation from Stockfish
        Uses existing engine instance when provided
        
        Args:
            board: chess.Board object
            engine: Optional chess.engine.SimpleEngine instance
            time_limit: Time limit for analysis in seconds
        
        Returns:
            float: Evaluation in pawns from current player's perspective
        """
        close_engine = False
        try:
            # Use provided engine or create new one
            if engine is None:
                engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
                close_engine = True
            
            # Set a very short time limit for quick analysis
            info = engine.analyse(board, chess.engine.Limit(time=time_limit))
            
            # Extract evaluation
            if "score" in info:
                score = info["score"].relative.score(mate_score=10000)
                # Convert to pawn units
                return score / 100.0
            return 0.0
        
        except Exception as e:
            print(f"Error in engine evaluation: {e}")
            return 0.0
        
        finally:
            # Only close if we created a new engine
            if close_engine and engine:
                engine.quit()
                
    def _to_gpu_arrays(self, X, y):
        """Convert numpy arrays to GPU arrays"""
        try:
            import cupy as cp
            X_gpu = cp.array(X, dtype=cp.float32)
            y_gpu = cudf.Series(y)
            return X_gpu, y_gpu
        except ImportError:
            print("CuPy not found. Install with: pip install cupy-cuda11x")
            return X, y
        except Exception as e:
            print(f"Error converting to GPU arrays: {e}")
            print("Falling back to CPU arrays")
            return X, y

    def train(self, X, y):
        """
        Train the regression model on the prepared data, utilizing GPU if available
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Evaluation scores
        """
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.use_gpu:
            try:
                print("Setting up GPU training pipeline...")

                # Convert data to GPU format
                import cupy as cp
                X_train_gpu = cp.array(X_train, dtype=cp.float32)
                X_val_gpu = cp.array(X_val, dtype=cp.float32)
                import cudf
                y_train_gpu = cudf.Series(y_train)
                y_val_gpu = cudf.Series(y_val)

                # Use SVR for regression instead of SVC
                from cuml.svm import SVR as cuSVR
                from cuml.preprocessing import StandardScaler as cuStandardScaler

                self.model = Pipeline([
                    ('scaler', cuStandardScaler()),
                    ('svr', cuSVR(kernel='rbf', C=1.0, epsilon=0.1))
                ])

                print("Training SVR model on GPU...")
                self.model.fit(X_train_gpu, y_train_gpu)

                # Evaluate on validation set
                from sklearn.metrics import mean_squared_error, r2_score
                val_pred = self.model.predict(X_val_gpu)
                mse = mean_squared_error(y_val_gpu.to_numpy(), val_pred.get())
                r2 = r2_score(y_val_gpu.to_numpy(), val_pred.get())
                print(f"Validation MSE (GPU): {mse:.4f}")
                print(f"Validation R² (GPU): {r2:.4f}")
                return r2

            except Exception as e:
                print(f"Error during GPU training: {e}")
                print("Falling back to CPU training")
                self.use_gpu = False

        # CPU fallback
        print("Setting up CPU training pipeline...")
        from sklearn.svm import SVR
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))
        ])
        print("Training SVR model on CPU...")
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        from sklearn.metrics import mean_squared_error, r2_score
        val_pred = self.model.predict(X_val)
        mse = mean_squared_error(y_val, val_pred)
        r2 = r2_score(y_val, val_pred)
        print(f"Validation MSE (CPU): {mse:.4f}")
        print(f"Validation R² (CPU): {r2:.4f}")
        return r2
    
    def evaluate_position(self, board):
        """
        Evaluate a chess position using the trained model
        
        Args:
            board (chess.Board): Chess position to evaluate
            
        Returns:
            float: Evaluation score in pawns (positive means white advantage)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        # Extract features
        features = self._extract_position_features(board)
        
        # Convert to appropriate format based on model type
        if self.use_gpu:
            try:
                import cupy as cp
                features_gpu = cp.array([features], dtype=cp.float32)
                evaluation = float(self.model.predict(features_gpu)[0])
            except Exception as e:
                # Fallback to CPU prediction if GPU fails
                print(f"GPU prediction failed: {e}, falling back to CPU")
                evaluation = float(self.model.predict([features])[0])
        else:
            # Make prediction on CPU
            evaluation = float(self.model.predict([features])[0])
            
        return evaluation
    
    def save_model(self, filename="chess_position_evaluator.pkl"):
        """
        Save the trained model to a file. If GPU was used, warns about compatibility.
        """
        if self.model is None:
            raise ValueError("No model to save")

        try:
            joblib.dump({
                "model": self.model,
                "use_gpu": self.use_gpu
            }, filename)
            print(f"Model saved to {filename} (GPU: {self.use_gpu})")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filename="chess_position_evaluator.pkl"):
        """
        Load a trained model and determine whether GPU should be used based on the model type.
        """
        try:
            data = joblib.load(filename)

            if isinstance(data, dict) and "model" in data:
                self.model = data["model"]
                was_gpu_trained = data.get("use_gpu", False)

                # Adjust use_gpu based on current environment
                if self.use_gpu and not was_gpu_trained:
                    print("Loaded a CPU-trained model. Disabling GPU inference for compatibility.")
                    self.use_gpu = False
                elif not self.use_gpu and was_gpu_trained:
                    print("Warning: This model was trained on GPU. Set use_gpu=True if compatible.")

            else:
                # Legacy fallback for plain model (no GPU info saved)
                self.model = data
                print("Model loaded (legacy format). Assuming CPU.")
                self.use_gpu = False

            print(f"Model loaded from {filename} (GPU: {self.use_gpu})")

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {filename}: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize the evaluator with GPU support
    evaluator = ChessPositionEvaluator(stockfish_path="/usr/games/stockfish", use_gpu=True)
    
    # Prepare data from PGN files
    pgn_files = ["standardover2000-2021.pgn"]  # Add your PGN files here
    X, y = evaluator.prepare_training_data(pgn_files, num_games=5000)  # Start with a small sample
    
    # Train the model on GPU
    evaluator.train(X, y)
    
    # Save the model
    evaluator.save_model("chess_position_evaluator_5000.pkl")
    
    # Example: Evaluate a position
    board = chess.Board()
    eval_score = evaluator.evaluate_position(board)
    print(f"Position evaluation: {eval_score:.2f} pawns")