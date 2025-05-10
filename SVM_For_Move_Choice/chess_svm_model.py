import chess
import chess.pgn
import chess.engine
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import os
import io
import joblib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class ChessMoveEvaluator:
    def __init__(self, stockfish_path="stockfish"):
        """
        Initialize the Chess Move Evaluator
        
        Args:
            stockfish_path (str): Path to Stockfish engine executable
        """
        self.stockfish_path = stockfish_path
        self.model = None
        self.scaler = None
        self.pst = self._generate_simple_pst()

    def _generate_simple_pst(self):
        """
        Create a very light-weight piece-square-table (PST).
    
        Each entry is the Manhattan distance of the square to the centre (d4-e5),
        normalised to [-1 , +1] and then multiplied by the classical material
        value of the piece.  White uses the table as-is, black uses the 8×8 mirror.
        Enough to give your SVM a sense of *where* a piece sits without hard-coding
        grand-master heuristics.
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
    
    def _material_on_board(self, board: chess.Board) -> int:
        vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9}
        total = 0
        for piece in board.piece_map().values():
            total += vals.get(piece.piece_type, 0) * (1 if piece.color else -1)
        return total

    def _extract_board_features(self, board):
        """
        Extract features from a chess board position
        
        Args:
            board (chess.Board): Current board position
            
        Returns:
            list: Features representing the board state
        """
        features = []
        
        # Material count
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
        
        #one-hot encoding board state
        vec = np.zeros(12 * 64, dtype=np.int8)
        for sq, piece in board.piece_map().items():
            offset = (piece.color * 6 + (piece.piece_type - 1)) * 64
            vec[offset + sq] = 1
        features.extend(vec)

        #pst scores
        w_pst, b_pst = self._pst_scores(board, self.pst)
        features.append(w_pst - b_pst)              # positional edge  (scalar)
        features.append(w_pst + b_pst)
        
        
        # Control of the center (e4, d4, e5, d5)
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        white_center_control = sum(1 for sq in center_squares if board.is_attacked_by(chess.WHITE, sq))
        black_center_control = sum(1 for sq in center_squares if board.is_attacked_by(chess.BLACK, sq))
        features.append(white_center_control - black_center_control)
        
        # Mobility (number of legal moves)
        if board.turn == chess.WHITE:
            features.append(len(list(board.legal_moves)))
            board.turn = chess.BLACK
            features.append(-len(list(board.legal_moves)))
            board.turn = chess.WHITE
        else:
            features.append(-len(list(board.legal_moves)))
            board.turn = chess.WHITE
            features.append(len(list(board.legal_moves)))
            board.turn = chess.BLACK
            
        # King safety (attacks near king)
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if king_square:
                king_neighborhood = chess.SquareSet(chess.BB_KING_ATTACKS[king_square])
                attacks = sum(1 for sq in king_neighborhood if board.is_attacked_by(not color, sq))
                if color == chess.WHITE:
                    features.append(-attacks)  # Negative for white king being attacked
                else:
                    features.append(attacks)  # Positive for black king being attacked
            else:
                features.append(0)  # No king (shouldn't happen in normal chess)
                
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
        
        features.append(black_doubled - white_doubled)  # Doubled pawns disadvantage
        
        # Game phase
        total_pieces = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK)) + \
                      len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK)) + \
                      len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK)) + \
                      len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK))
        features.append(total_pieces)  # Higher means earlier game phases
        
        # Check status
        features.append(1 if board.is_check() else 0)
        
        return features
        
    def _extract_move_features(self, board: chess.Board, move: chess.Move) -> list[float]:
        feat = []

        moving_piece = board.piece_at(move.from_square)
        captured_piece = board.piece_at(move.to_square)

        moving_piece = board.piece_at(move.from_square)
        if moving_piece:
            piece_type_one_hot = [0] * 6  # One-hot encoding for piece type
            piece_type_one_hot[moving_piece.piece_type - 1] = 1  # Adjust for 0-indexing
            feat.extend(piece_type_one_hot)
        else:
            feat.extend([0, 0, 0, 0, 0, 0])  # Shouldn't happen with legal moves

        to_square = move.to_square
        feat.append(1 if board.is_attacked_by(not board.turn, to_square) else 0)

        # 1. Value of moving piece  -----------------------------------------
        feat.append(self._get_piece_value(moving_piece.piece_type))

        # 2. Manhattan distance travelled  ----------------------------------
        feat.append(self._manhattan(move.from_square, move.to_square))

        # 3. Is capture (binary)  -------------------------------------------
        feat.append(1 if captured_piece else 0)

        # 4. Static exchange evaluation (SEE lite)  -------------------------
        see = 0
        if captured_piece:
            see = self._get_piece_value(captured_piece.piece_type) - \
                self._get_piece_value(moving_piece.piece_type)
        feat.append(see)

        # 5. Δ material after the move  -------------------------------------
        before = self._material_on_board(board)
        board_copy = board.copy()
        board_copy.push(move)
        after = self._material_on_board(board_copy)
        feat.append(after - before)

        # 6. Promotion piece value (0 if none)  -----------------------------
        feat.append(self._get_piece_value(move.promotion) if move.promotion else 0)

        # 7. Castling (binary)  ---------------------------------------------
        feat.append(1 if board.is_castling(move) else 0)

        # Check if move gives check
        board_copy = board.copy()
        board_copy.push(move)
        feat.append(1 if board_copy.is_check() else 0)

        return feat
            

    def prepare_training_data(self, pgn_files, num_games=None):
        """
        Process PGN files to create training data - optimized version
        Args:
            pgn_files (list): List of PGN file paths
            num_games (int): Maximum number of games to process (None for all)
        Returns:
            tuple: (X, y) feature matrix and target values
        """
        
        
        features_list = []
        labels = []
        games_processed = 0
        
        # Batch size for processing
        BATCH_SIZE = 10  # Process 10 positions at once before sending to engine
        
        # Create engine once outside of game loop
        with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
            # Configure engine for faster analysis
            engine.configure({"Threads": 4, "Hash": 128})  # Adjust based on your system
            
            # Create a position evaluation cache to avoid re-evaluating same positions
            position_cache = {}
            
            for pgn_file in pgn_files:
                print(f"Processing {pgn_file}...")
                
                # Load games into memory first
                games = []
                with open(pgn_file) as pgn:
                    while True:
                        game = chess.pgn.read_game(pgn)
                        if game is None:
                            break
                        games.append(game)
                        games_processed += 1
                        if num_games is not None and games_processed >= num_games:
                            break
                
                # Process loaded games
                for game_idx, game in enumerate(games):
                    print(f"Processing game {game_idx+1} of {len(games)}")
                    board = game.board()
                    
                    # Collect positions and moves
                    positions = []
                    moves = []
                    for move in game.mainline_moves():
                        positions.append(board.copy())
                        moves.append(move)
                        board.push(move)
                    
                    # Process in batches for faster evaluation
                    for i in range(0, len(positions), BATCH_SIZE):
                        batch_positions = positions[i:i+BATCH_SIZE]
                        batch_moves = moves[i:i+BATCH_SIZE]
                        
                        # Extract board features for all positions in batch (can be done in parallel)
                        board_features_batch = [self._extract_board_features(pos) for pos in batch_positions]
                        
                        # Extract move features for all moves in batch
                        move_features_batch = [self._extract_move_features(batch_positions[j], batch_moves[j]) 
                                            for j in range(len(batch_positions))]
                        
                        # Get evaluations for all positions in batch
                        evals_before = []
                        for pos in batch_positions:
                            pos_hash = pos.fen()
                            if pos_hash in position_cache:
                                evals_before.append(position_cache[pos_hash])
                            else:
                                eval_score = self._get_stockfish_evaluation(pos, engine)
                                position_cache[pos_hash] = eval_score
                                evals_before.append(eval_score)
                        
                        # Get evaluations after moves
                        evals_after = []
                        for j, (pos, move) in enumerate(zip(batch_positions, batch_moves)):
                            # Make move
                            new_pos = pos.copy()
                            new_pos.push(move)
                            pos_hash = new_pos.fen()
                            
                            if pos_hash in position_cache:
                                # Negate because perspective switches
                                evals_after.append(-position_cache[pos_hash])
                            else:
                                eval_score = self._get_stockfish_evaluation(new_pos, engine)
                                position_cache[pos_hash] = eval_score
                                # Negate because perspective switches
                                evals_after.append(-eval_score)
                        
                        # Process results for this batch
                        for j in range(len(batch_positions)):
                            # Create combined feature vector
                            all_features = board_features_batch[j] + move_features_batch[j]
                            
                            # If eval improved, this is a "good" move (class 1), otherwise "bad" (class 0)
                            improved = (evals_after[j] - evals_before[j]) > 0.1
                            
                            features_list.append(all_features)
                            labels.append(1 if improved else 0)
        
        return np.array(features_list), np.array(labels)

    # Updated Stockfish evaluation method that accepts an engine parameter
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
        
    def train(self, X, y):
        """
        Train the SVM model on the prepared data
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target labels
        """
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create a pipeline with scaling and SVM
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', C=1.0, probability=True))
        ])
        
        # Train the model
        print("Training SVM model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_accuracy = self.model.score(X_val, y_val)
        print(f"Validation accuracy: {val_accuracy:.4f}")
        
        return val_accuracy
    
    def evaluate_move(self, board, move):
        """
        Evaluate a candidate move using the trained model
        
        Args:
            board (chess.Board): Current board position
            move (chess.Move): Candidate move to evaluate
            
        Returns:
            float: Confidence score between 0 and 1
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        # Extract features
        board_features = self._extract_board_features(board)
        move_features = self._extract_move_features(board, move)
        all_features = board_features + move_features
        
        # Make prediction
        confidence = self.model.predict_proba([all_features])[0][1]  # Probability of class 1
        return confidence
    
    def find_best_move(self, board):
        """
        Find the best move in the current position according to the model
        
        Args:
            board (chess.Board): Current board position
            
        Returns:
            tuple: (best_move, confidence) the best move and its confidence score
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        best_move = None
        best_confidence = -1
        
        for move in board.legal_moves:
            confidence = self.evaluate_move(board, move)
            if confidence > best_confidence:
                best_confidence = confidence
                best_move = move
                
        return best_move, best_confidence
    
    def save_model(self, filename="chess_svm_model.pkl"):
        """Save the trained model to a file"""
        if self.model is None:
            raise ValueError("No model to save")
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename="chess_svm_model.pkl"):
        """Load a trained model from a file"""
        self.model = joblib.load(filename)
        print(f"Model loaded from {filename}")


# Example usage
if __name__ == "__main__":
    # Initialize the evaluator
    evaluator = ChessMoveEvaluator(stockfish_path="/usr/games/stockfish")  # Update path as needed
    
    # Prepare data from PGN files
    pgn_files = ["standard16klinesover2000.pgn"]  # Add your PGN files here
    X, y = evaluator.prepare_training_data(pgn_files, num_games=700)  # Start with a small sample
    
    # Train the model
    evaluator.train(X, y)
    
    # Save the model
    evaluator.save_model("chess_svm_model_700_games_featuresV3.pkl")
    
    # Example: Evaluate a position
    board = chess.Board()
    best_move, confidence = evaluator.find_best_move(board)
    print(f"Best move: {best_move}, Confidence: {confidence:.4f}")