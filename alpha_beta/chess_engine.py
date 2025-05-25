import chess
import time
import random
from svm_position_evaluator import ChessPositionEvaluator

# class ChessEngine:
#     """
#     A simple chess engine that uses alpha-beta pruning and a position evaluator model
#     to select the best moves.
#     """
    
#     def __init__(self, position_evaluator=None, stockfish_path=None, search_depth=3):
#         """
#         Initialize the chess engine.
        
#         Args:
#             position_evaluator: A trained position evaluation model
#             stockfish_path: Path to Stockfish engine (for fallback evaluation)
#             search_depth: Default depth for alpha-beta search
#         """
#         self.evaluator = position_evaluator
#         self.stockfish_path = stockfish_path
#         self.default_depth = search_depth
#         self.nodes_evaluated = 0
#         self.tt_table = {}  # Transposition table for storing evaluated positions
        
#         # If no evaluator is provided, create a default one
#         if self.evaluator is None and stockfish_path is not None:
#             self.evaluator = ChessPositionEvaluator(stockfish_path=stockfish_path)
#             print("Warning: Using untrained position evaluator")
    
#     def evaluate_position(self, board):
#         """
#         Evaluate the current position using the trained model.
        
#         Args:
#             board: A chess.Board object representing the current position
        
#         Returns:
#             float: Score from white's perspective in centipawns
#         """
#         # Use our trained model if available
#         if self.evaluator and self.evaluator.model is not None:
#             try:
#                 return self.evaluator.evaluate_position(board) * 100  # Convert to centipawns
#             except Exception as e:
#                 print(f"Model evaluation failed: {e}")
        
#         # Fallback to a simple material evaluator
#         return self._simple_evaluate(board)
    
#     def get_best_move(self, board, depth=None, time_limit=None):
#         """
#         Find the best move in the current position using alpha-beta search.
        
#         Args:
#             board: A chess.Board object
#             depth: Search depth (if None, use default_depth)
#             time_limit: Time limit in seconds (optional)
            
#         Returns:
#             chess.Move: The best move found
#         """
#         if depth is None:
#             depth = self.default_depth
            
#         self.nodes_evaluated = 0
#         self.tt_table = {}  # Clear transposition table
#         start_time = time.time()
        
#         # Use iterative deepening if time limit is provided
#         if time_limit is not None:
#             return self._iterative_deepening(board, depth, time_limit)
            
#         alpha = float('-inf')
#         beta = float('inf')
#         best_move = None
#         best_score = float('-inf')
        
#         # Get all legal moves
#         legal_moves = list(board.legal_moves)
        
#         # Move ordering: try captures first
#         legal_moves.sort(key=lambda move: self._move_score(board, move), reverse=True)
        
#         for move in legal_moves:
#             board.push(move)
            
#             # Use negamax with alpha-beta pruning
#             score = -self._alpha_beta(board, depth - 1, -beta, -alpha)
            
#             board.pop()
            
#             if score > best_score:
#                 best_score = score
#                 best_move = move
            
#             alpha = max(alpha, score)
        
#         elapsed = time.time() - start_time
#         print(f"Evaluated {self.nodes_evaluated} nodes in {elapsed:.2f}s ({self.nodes_evaluated/elapsed:.0f} nps)")
#         print(f"Best move: {board.san(best_move)} with score: {best_score/100:.2f}")
        
#         return best_move
    
#     def _move_score(self, board, move):
#         """
#         Heuristic score for move ordering.
#         Higher scores will be searched first.
        
#         Args:
#             board: A chess.Board object
#             move: A chess.Move object
            
#         Returns:
#             float: Heuristic score for the move
#         """
#         score = 0
        
#         # Captures are scored by MVV-LVA (Most Valuable Victim, Least Valuable Aggressor)
#         piece_values = {
#             chess.PAWN: 1,
#             chess.KNIGHT: 3,
#             chess.BISHOP: 3,
#             chess.ROOK: 5,
#             chess.QUEEN: 9,
#             chess.KING: 0  # Special case, handled below
#         }
        
#         # Bonus for captures (MVV-LVA)
#         if board.is_capture(move):
#             target_square = move.to_square
#             target_piece = board.piece_at(target_square)
            
#             if target_piece:
#                 # Value of captured piece
#                 victim_value = piece_values.get(target_piece.piece_type, 0) * 100
                
#                 # Value of attacking piece
#                 aggressor = board.piece_at(move.from_square)
#                 aggressor_value = piece_values.get(aggressor.piece_type, 1)
                
#                 # MVV-LVA score = victim value - aggressor value/10
#                 # This prioritizes capturing high-value pieces with low-value attackers
#                 score += victim_value - aggressor_value * 10
        
#         # Bonus for promotions
#         if move.promotion:
#             score += piece_values.get(move.promotion, 0) * 90
        
#         # Bonus for attacking center in the opening
#         if board.fullmove_number <= 10:
#             center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
#             if move.to_square in center_squares:
#                 score += 30
        
#         # Bonus for castling
#         if board.is_castling(move):
#             score += 50
        
#         # Bonus for checks
#         board.push(move)
#         if board.is_check():
#             score += 20
#         board.pop()
        
#         return score
    
#     def _iterative_deepening(self, board, max_depth, time_limit):
#         """
#         Perform iterative deepening search up to max_depth with a time limit.
        
#         Args:
#             board: A chess.Board object
#             max_depth: Maximum search depth
#             time_limit: Time limit in seconds
            
#         Returns:
#             chess.Move: The best move found within the time limit
#         """
#         start_time = time.time()
#         best_move = None
#         last_score = 0
        
#         # Start with depth 1 and increase gradually
#         for depth in range(1, max_depth + 1):
#             alpha = float('-inf')
#             beta = float('inf')
#             best_score = float('-inf')
#             current_best_move = None
            
#             legal_moves = list(board.legal_moves)
            
#             # Use the best move from previous iteration first (if available)
#             if best_move in legal_moves:
#                 legal_moves.remove(best_move)
#                 legal_moves.insert(0, best_move)
#             else:
#                 legal_moves.sort(key=lambda move: self._move_score(board, move), reverse=True)
            
#             for move in legal_moves:
#                 board.push(move)
                
#                 # Use negamax with alpha-beta pruning
#                 score = -self._alpha_beta(board, depth - 1, -beta, -alpha)
                
#                 board.pop()
                
#                 if score > best_score:
#                     best_score = score
#                     current_best_move = move
                
#                 alpha = max(alpha, score)
                
#                 # Check if we're out of time
#                 if time.time() - start_time > time_limit:
#                     print(f"Reached time limit at depth {depth}")
#                     return best_move if best_move is not None else current_best_move
            
#             # Update best move after completing a full depth
#             best_move = current_best_move
            
#             # Log progress
#             elapsed = time.time() - start_time
#             print(f"Depth {depth} completed in {elapsed:.2f}s, best move: {board.san(best_move)}, score: {best_score/100:.2f}")
            
#             # Score isn't changing significantly between iterations, we might have reached a stable evaluation
#             if abs(best_score - last_score) < 10 and depth >= 3 and elapsed > time_limit * 0.75:
#                 print(f"Stable evaluation reached, stopping at depth {depth}")
#                 break
                
#             last_score = best_score
            
#             # Check if we're out of time before starting the next depth
#             if time.time() - start_time > time_limit * 0.8:  # Leave a small buffer
#                 print(f"Time limit approaching, stopping at depth {depth}")
#                 break
                
#         return best_move
    
#     def _alpha_beta(self, board, depth, alpha, beta):
#         """
#         Alpha-beta pruning implementation with negamax framework.
        
#         Args:
#             board: A chess.Board object
#             depth: Remaining search depth
#             alpha: Alpha value for pruning
#             beta: Beta value for pruning
            
#         Returns:
#             float: Evaluation score from the perspective of the current player
#         """
#         self.nodes_evaluated += 1
        
#         # Check for immediate game end
#         if board.is_game_over():
#             if board.is_checkmate():
#                 # Return a large negative score if current player is checkmated
#                 return -9900  # Slightly less than max value to prefer quicker mates
#             else:
#                 # Draw
#                 return 0
        
#         # Check transposition table
#         board_hash = self._get_board_hash(board)
#         if board_hash in self.tt_table and self.tt_table[board_hash]['depth'] >= depth:
#             return self.tt_table[board_hash]['score']
        
#         # Quiescence search at leaf nodes to handle horizon effect
#         if depth == 0:
#             return self._quiescence_search(board, alpha, beta)
        
#         # Get legal moves and sort them
#         legal_moves = list(board.legal_moves)
#         legal_moves.sort(key=lambda move: self._move_score(board, move), reverse=True)
        
#         best_score = float('-inf')
        
#         for move in legal_moves:
#             board.push(move)
#             score = -self._alpha_beta(board, depth - 1, -beta, -alpha)
#             board.pop()
            
#             best_score = max(best_score, score)
#             alpha = max(alpha, score)
            
#             # Alpha-beta cutoff
#             if alpha >= beta:
#                 break
        
#         # Store in transposition table
#         self.tt_table[board_hash] = {'depth': depth, 'score': best_score}
#         return best_score
        
#     def _quiescence_search(self, board, alpha, beta, max_depth=3):
#         """
#         Quiescence search to handle the horizon effect.
#         Continues searching captures to reach a "quiet" position.
        
#         Args:
#             board: A chess.Board object
#             alpha: Alpha value for pruning
#             beta: Beta value for pruning
#             max_depth: Maximum additional depth for quiescence search
            
#         Returns:
#             float: Evaluation score from the perspective of the current player
#         """
#         self.nodes_evaluated += 1
        
#         # Stand pat - evaluate current position
#         stand_pat = self.evaluate_position(board)
#         stand_pat = stand_pat if board.turn == chess.WHITE else -stand_pat
        
#         if stand_pat >= beta:
#             return beta
        
#         alpha = max(alpha, stand_pat)
        
#         # If we've gone too deep in quiescence, return evaluation
#         if max_depth <= 0:
#             return stand_pat
        
#         # Look at capture moves only
#         captures = [move for move in board.legal_moves if board.is_capture(move)]
#         captures.sort(key=lambda move: self._move_score(board, move), reverse=True)
        
#         for move in captures:
#             board.push(move)
#             score = -self._quiescence_search(board, -beta, -alpha, max_depth - 1)
#             board.pop()
            
#             if score >= beta:
#                 return beta
            
#             alpha = max(alpha, score)
        
#         return alpha
    
#     def _is_capture(self, board, move):
#         """
#         Check if a move is a capture for move ordering.
        
#         Args:
#             board: A chess.Board object
#             move: A chess.Move object
            
#         Returns:
#             bool: True if the move is a capture
#         """
#         return board.is_capture(move)
    
#     def _get_board_hash(self, board):
#         """
#         Get a hash of the current board position for the transposition table.
        
#         Args:
#             board: A chess.Board object
            
#         Returns:
#             str: A string hash of the board position
#         """
#         return board.fen()


# def play_game(engine, initial_position=None, max_moves=100):
#     """
#     Play a complete game using the chess engine.
    
#     Args:
#         engine: A ChessEngine instance
#         initial_position: Optional starting FEN string
#         max_moves: Maximum number of moves before forcing a draw
        
#     Returns:
#         chess.Board: Final board position
#         str: Game result
#     """
#     if initial_position:
#         board = chess.Board(initial_position)
#     else:
#         board = chess.Board()
    
#     move_count = 0
    
#     while not board.is_game_over() and move_count < max_moves:
#         print(f"\nMove {move_count + 1}, {'White' if board.turn == chess.WHITE else 'Black'} to move")
#         print(board)
        
#         # Use engine to find best move
#         start_time = time.time()
#         move = engine.get_best_move(board, time_limit=3.0)  # 3 seconds per move
#         elapsed = time.time() - start_time
        
#         print(f"Engine chose {board.san(move)} in {elapsed:.2f}s")
#         board.push(move)
#         move_count += 1
    
#     print("\nFinal position:")
#     print(board)
    
#     # Determine result
#     if board.is_checkmate():
#         result = "1-0" if board.turn == chess.BLACK else "0-1"
#         print(f"Checkmate! {'White' if result == '1-0' else 'Black'} wins.")
#     elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
#         result = "1/2-1/2"
#         print("Draw!")
#     elif move_count >= max_moves:
#         result = "1/2-1/2"
#         print(f"Game drawn after {max_moves} moves.")
#     else:
#         result = "Unknown"
    
#     return board, result


# if __name__ == "__main__":
#     # Example usage
#     model_path = "chess_position_evaluator_5000.pkl"
#     stockfish_path = "/usr/games/stockfish"
    
#     # Initialize position evaluator and load model
#     evaluator = ChessPositionEvaluator(stockfish_path=stockfish_path)
#     try:
#         evaluator.load_model(model_path)
#         print(f"Successfully loaded model from {model_path}")
#     except Exception as e:
#         print(f"Failed to load model: {e}")
#         print("Will use fallback evaluator")
#         evaluator = None
    
#     # Create chess engine
#     engine = ChessEngine(position_evaluator=evaluator, stockfish_path=stockfish_path, search_depth=4)
    
#     # Example - find best move in current position
#     board = chess.Board()
#     print("Initial position:")
#     print(board)
    
#     best_move = engine.get_best_move(board, time_limit=5.0)
#     print(f"Best move: {board.san(best_move)}")
    
#     # Try with a more interesting position
#     fen = "r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
#     board = chess.Board(fen)
#     print("\nInteresting position (Italian Game):")
#     print(board)
    
#     best_move = engine.get_best_move(board, time_limit=5.0)
#     print(f"Best move: {board.san(best_move)}")
    
#     # Uncomment to play a complete game
#     # play_game(engine)


class ParallelPositionEvaluator:
    """
    A wrapper for parallel evaluation of chess positions using multiple evaluator instances.
    """
    
    def __init__(self, model_path, stockfish_path, num_evaluators=4, use_gpu=True):
        """
        Initialize parallel position evaluator.
        
        Args:
            model_path: Path to the trained model file
            stockfish_path: Path to Stockfish engine
            num_evaluators: Number of parallel evaluators to create
            use_gpu: Whether to use GPU for evaluation
        """
        self.evaluators = []
        self.num_evaluators = num_evaluators
        self.model_path = model_path
        self.stockfish_path = stockfish_path
        self.use_gpu = use_gpu
        
        # Create separate evaluators
        for i in range(num_evaluators):
            evaluator = ChessPositionEvaluator(stockfish_path=stockfish_path, use_gpu=use_gpu)
            try:
                evaluator.load_model(model_path)
                self.evaluators.append(evaluator)
            except Exception as e:
                print(f"Failed to load model for evaluator {i}: {e}")
        
        if not self.evaluators:
            raise ValueError("Failed to initialize any evaluator")
        
        print(f"Initialized {len(self.evaluators)} parallel position evaluators")
        
        # Set up thread pool for parallel evaluations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_evaluators)
        
        # Set up evaluation queue for batch processing
        self.eval_queue = queue.Queue()
        self.eval_results = {}
        self.batch_size = 64#  # Process positions in batches for better GPU utilization
        self.eval_lock = threading.Lock()
        
        # Start the evaluation worker thread
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._evaluation_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def evaluate_position(self, board):
        """
        Evaluate a chess position using the trained model.
        
        Args:
            board: A chess.Board object representing the position
            
        Returns:
            float: Evaluation score in centipawns from white's perspective
        """
        # Use the first evaluator directly for single evaluations
        return self.evaluators[0].evaluate_position(board) * 100
    
    def evaluate_positions_batch(self, boards):
        """
        Evaluate a batch of positions in parallel.
        
        Args:
            boards: List of chess.Board objects
            
        Returns:
            list: List of evaluation scores in centipawns
        """
        if not boards:
            return []
        
        # Submit all positions to the thread pool
        future_to_board = {
            self.executor.submit(
                self.evaluators[i % len(self.evaluators)].evaluate_position, 
                board
            ): i for i, board in enumerate(boards)
        }
        
        # Collect results
        results = [None] * len(boards)
        for future in concurrent.futures.as_completed(future_to_board):
            index = future_to_board[future]
            try:
                result = future.result()
                results[index] = result * 100  # Convert to centipawns
            except Exception as e:
                print(f"Evaluation failed: {e}")
                results[index] = 0
        
        return results
    
    def _evaluation_worker(self):
        """
        Worker thread that processes the evaluation queue in batches.
        """
        batch = []
        batch_ids = []
        
        while not self.stop_event.is_set():
            try:
                # Get the next position from the queue with timeout
                try:
                    board_id, board = self.eval_queue.get(timeout=0.1)
                    batch.append(board)
                    batch_ids.append(board_id)
                except queue.Empty:
                    # Queue is empty, process current batch if it exists
                    if batch:
                        self._process_batch(batch, batch_ids)
                        batch = []
                        batch_ids = []
                    continue
                
                # Process a batch if it's full
                if len(batch) >= self.batch_size:
                    self._process_batch(batch, batch_ids)
                    batch = []
                    batch_ids = []
                    
            except Exception as e:
                print(f"Error in evaluation worker: {e}")
    
    def _process_batch(self, batch, batch_ids):
        """
        Process a batch of positions.
        
        Args:
            batch: List of chess.Board objects
            batch_ids: List of IDs corresponding to the boards
        """
        # Evaluate the batch
        results = self.evaluate_positions_batch(batch)
        
        # Store results
        with self.eval_lock:
            for board_id, result in zip(batch_ids, results):
                self.eval_results[board_id] = result
    
    def submit_position(self, board_id, board):
        """
        Submit a position for asynchronous evaluation.
        
        Args:
            board_id: Unique identifier for this position
            board: chess.Board object
        """
        self.eval_queue.put((board_id, board.copy()))
    
    def get_result(self, board_id):
        """
        Get the result of an evaluation.
        
        Args:
            board_id: ID of the position
            
        Returns:
            float: Evaluation score if available, None otherwise
        """
        with self.eval_lock:
            return self.eval_results.get(board_id)
    
    def shutdown(self):
        """Clean up resources."""
        self.stop_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        self.executor.shutdown(wait=False)
        
import chess
import time
import random
import concurrent.futures
import threading
import queue
from collections import defaultdict
from svm_position_evaluator import ChessPositionEvaluator

class ChessEngine:
    """
    A simple chess engine that uses alpha-beta pruning and a position evaluator model
    to select the best moves.
    """
    
    def __init__(self, position_evaluator=None, stockfish_path=None, search_depth=3, num_parallel_evaluators=4):
        """
        Initialize the chess engine.
        
        Args:
            position_evaluator: A trained position evaluation model or ParallelPositionEvaluator
            stockfish_path: Path to Stockfish engine (for fallback evaluation)
            search_depth: Default depth for alpha-beta search
            num_parallel_evaluators: Number of evaluators to use in parallel
        """
        self.default_depth = search_depth
        self.nodes_evaluated = 0
        self.tt_table = {}  # Transposition table for storing evaluated positions
        self.stockfish_path = stockfish_path
        self.parallel_evaluator = None
        
        # Track positions waiting for evaluation
        self.pending_evaluations = {}
        self.next_eval_id = 0
        
        # Set up the evaluator
        if isinstance(position_evaluator, ParallelPositionEvaluator):
            self.parallel_evaluator = position_evaluator
            self.evaluator = None
        elif position_evaluator is not None:
            self.evaluator = position_evaluator
            self.parallel_evaluator = None
        elif stockfish_path is not None:
            # Create a parallel evaluator if none was provided
            try:
                model_path = "chess_position_evaluator_5000.pkl"  # Assuming default path
                self.parallel_evaluator = ParallelPositionEvaluator(
                    model_path=model_path,
                    stockfish_path=stockfish_path, 
                    num_evaluators=num_parallel_evaluators
                )
                self.evaluator = None
            except Exception as e:
                print(f"Failed to create parallel evaluator: {e}")
                print("Using single evaluator as fallback")
                self.evaluator = ChessPositionEvaluator(stockfish_path=stockfish_path)
                try:
                    self.evaluator.load_model(model_path)
                except Exception:
                    print("Warning: Using untrained position evaluator")
                self.parallel_evaluator = None
        else:
            self.evaluator = None
            self.parallel_evaluator = None
            print("Warning: No evaluator provided, using simple material evaluation")
    
    def evaluate_position(self, board):
        """
        Evaluate the current position using the trained model.
        
        Args:
            board: A chess.Board object representing the current position
        
        Returns:
            float: Score from white's perspective in centipawns
        """
        # Use parallel evaluator if available
        if self.parallel_evaluator is not None:
            try:
                return self.parallel_evaluator.evaluate_position(board)
            except Exception as e:
                print(f"Parallel evaluation failed: {e}")
        
        # Use single evaluator as fallback
        if self.evaluator and self.evaluator.model is not None:
            try:
                return self.evaluator.evaluate_position(board) * 100  # Convert to centipawns
            except Exception as e:
                print(f"Model evaluation failed: {e}")
        
        # Fallback to a simple material evaluator
        return self._simple_evaluate(board)
    
    def _simple_evaluate(self, board):
        """
        A simple material-based evaluation function as fallback.
        
        Args:
            board: A chess.Board object
            
        Returns:
            float: Score from white's perspective in centipawns
        """
        if board.is_checkmate():
            # Return a large negative score if white is checkmated, or a large positive score if black is checkmated
            return -10000 if board.turn == chess.WHITE else 10000
        
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return 0  # Draw
        
        # Piece values in centipawns (1 pawn = 100 cp)
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Calculate material balance
        white_material = sum(len(board.pieces(piece_type, chess.WHITE)) * value 
                            for piece_type, value in piece_values.items())
        black_material = sum(len(board.pieces(piece_type, chess.BLACK)) * value
                            for piece_type, value in piece_values.items())
        
        # Return score from white's perspective
        score = white_material - black_material
        
        # Add position-based bonuses
        score += self._positional_bonuses(board)
        
        return score
        
    def _positional_bonuses(self, board):
        """
        Calculate positional bonuses to make the evaluation more realistic.
        
        Args:
            board: A chess.Board object
            
        Returns:
            float: Positional score bonus in centipawns
        """
        score = 0
        
        # Center control bonus for pawns and knights
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        center_control = 20  # centipawns
        
        for square in center_squares:
            if board.piece_at(square):
                piece = board.piece_at(square)
                if piece.piece_type in [chess.PAWN, chess.KNIGHT]:
                    # Add bonus for white, subtract for black
                    score += center_control if piece.color == chess.WHITE else -center_control
        
        # Development bonus in the opening
        if board.fullmove_number <= 10:
            # Bonus for developed knights and bishops
            knight_development = 15
            bishop_development = 15
            
            # Knights developed from original squares
            if not board.piece_at(chess.B1) and board.piece_type_at(chess.G1) != chess.KNIGHT:
                score += knight_development  # White queenside knight moved
            if not board.piece_at(chess.G1) and board.piece_type_at(chess.B1) != chess.KNIGHT:
                score += knight_development  # White kingside knight moved
            if not board.piece_at(chess.B8) and board.piece_type_at(chess.G8) != chess.KNIGHT:
                score -= knight_development  # Black queenside knight moved
            if not board.piece_at(chess.G8) and board.piece_type_at(chess.B8) != chess.KNIGHT:
                score -= knight_development  # Black kingside knight moved
            
            # Bishops developed from original squares
            if not board.piece_at(chess.C1) and board.piece_type_at(chess.F1) != chess.BISHOP:
                score += bishop_development  # White queenside bishop moved
            if not board.piece_at(chess.F1) and board.piece_type_at(chess.C1) != chess.BISHOP:
                score += bishop_development  # White kingside bishop moved
            if not board.piece_at(chess.C8) and board.piece_type_at(chess.F8) != chess.BISHOP:
                score -= bishop_development  # Black queenside bishop moved
            if not board.piece_at(chess.F8) and board.piece_type_at(chess.C8) != chess.BISHOP:
                score -= bishop_development  # Black kingside bishop moved
        
        # Castling bonus
        castling_bonus = 40
        if board.has_castling_rights(chess.WHITE):
            if not board.has_kingside_castling_rights(chess.WHITE) and not board.has_queenside_castling_rights(chess.WHITE):
                # White has no castling rights but had them (either already castled or lost them)
                if board.king(chess.WHITE) in [chess.G1, chess.C1]:
                    score += castling_bonus  # White has castled
        if board.has_castling_rights(chess.BLACK):
            if not board.has_kingside_castling_rights(chess.BLACK) and not board.has_queenside_castling_rights(chess.BLACK):
                # Black has no castling rights but had them (either already castled or lost them)
                if board.king(chess.BLACK) in [chess.G8, chess.C8]:
                    score -= castling_bonus  # Black has castled
        
        # King safety in the middlegame/endgame
        # Simple estimate based on pieces around the king
        if board.fullmove_number > 10:
            white_king_sq = board.king(chess.WHITE)
            black_king_sq = board.king(chess.BLACK)
            
            white_king_safety = 0
            black_king_safety = 0
            
            # Check squares around the king
            for offset in [8, -8, 1, -1, 9, -9, 7, -7]:  # Adjacent squares
                # White king safety
                adjacent_sq = white_king_sq + offset
                try:
                    if 0 <= adjacent_sq < 64:  # Valid square
                        piece = board.piece_at(adjacent_sq)
                        if piece and piece.color == chess.WHITE:
                            white_king_safety += 5  # Friendly piece near king
                except ValueError:
                    pass
                
                # Black king safety
                adjacent_sq = black_king_sq + offset
                try:
                    if 0 <= adjacent_sq < 64:  # Valid square
                        piece = board.piece_at(adjacent_sq)
                        if piece and piece.color == chess.BLACK:
                            black_king_safety += 5  # Friendly piece near king
                except ValueError:
                    pass
            
            score += white_king_safety - black_king_safety
        
        return score
    
    def get_best_move(self, board, depth=None, time_limit=None):
        """
        Find the best move in the current position using alpha-beta search.
        
        Args:
            board: A chess.Board object
            depth: Search depth (if None, use default_depth)
            time_limit: Time limit in seconds (optional)
            
        Returns:
            chess.Move: The best move found
        """
        if depth is None:
            depth = self.default_depth
            
        self.nodes_evaluated = 0
        self.tt_table = {}  # Clear transposition table
        self.pending_evaluations = {}
        self.next_eval_id = 0
        
        start_time = time.time()
        
        # Use iterative deepening if time limit is provided
        if time_limit is not None:
            return self._iterative_deepening(board, depth, time_limit)
            
        alpha = float('-inf')
        beta = float('inf')
        best_move = None
        best_score = float('-inf')
        
        # Get all legal moves
        legal_moves = list(board.legal_moves)
        
        # Move ordering: try captures first
        legal_moves.sort(key=lambda move: self._move_score(board, move), reverse=True)
        
        # If we have a parallel evaluator, pre-evaluate all positions from the first ply
        # to get better move ordering
        if self.parallel_evaluator and len(legal_moves) > 3:
            print("Pre-evaluating first-ply positions for better move ordering...")
            first_ply_boards = []
            for move in legal_moves:
                new_board = board.copy()
                new_board.push(move)
                first_ply_boards.append(new_board)
            
            # Evaluate in parallel
            scores = self.parallel_evaluator.evaluate_positions_batch(first_ply_boards)
            
            # Sort moves by evaluation score (best first)
            move_scores = [(move, -scores[i] if board.turn == chess.WHITE else scores[i]) 
                           for i, move in enumerate(legal_moves)]
            move_scores.sort(key=lambda x: x[1], reverse=True)
            legal_moves = [move for move, _ in move_scores]
        
        futures = []
        if self.parallel_evaluator:
            # For parallel searching at the root level
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(legal_moves)) as executor:
                for move in legal_moves:
                    # Make a copy of the board for each thread
                    board_copy = board.copy()
                    board_copy.push(move)
                    
                    # Submit the search task
                    future = executor.submit(
                        self._alpha_beta, 
                        board_copy, 
                        depth - 1, 
                        -beta, 
                        -alpha
                    )
                    futures.append((move, future))
                
                # Collect results as they complete
                for move, future in futures:
                    try:
                        score = -future.result()
                        
                        if score > best_score:
                            best_score = score
                            best_move = move
                        
                        alpha = max(alpha, score)
                    except Exception as e:
                        print(f"Error in parallel search: {e}")
        else:
            # Sequential search
            for move in legal_moves:
                board.push(move)
                
                # Use negamax with alpha-beta pruning
                score = -self._alpha_beta(board, depth - 1, -beta, -alpha)
                
                board.pop()
                
                if score > best_score:
                    best_score = score
                    best_move = move
                
                alpha = max(alpha, score)
        
        elapsed = time.time() - start_time
        print(f"Evaluated {self.nodes_evaluated} nodes in {elapsed:.2f}s ({self.nodes_evaluated/elapsed:.0f} nps)")
        print(f"Best move: {board.san(best_move)} with score: {best_score/100:.2f}")
        
        return best_move
    
    def _move_score(self, board, move):
        """
        Heuristic score for move ordering.
        Higher scores will be searched first.
        
        Args:
            board: A chess.Board object
            move: A chess.Move object
            
        Returns:
            float: Heuristic score for the move
        """
        score = 0
        
        # Captures are scored by MVV-LVA (Most Valuable Victim, Least Valuable Aggressor)
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # Special case, handled below
        }
        
        # Bonus for captures (MVV-LVA)
        if board.is_capture(move):
            target_square = move.to_square
            target_piece = board.piece_at(target_square)
            
            if target_piece:
                # Value of captured piece
                victim_value = piece_values.get(target_piece.piece_type, 0) * 100
                
                # Value of attacking piece
                aggressor = board.piece_at(move.from_square)
                aggressor_value = piece_values.get(aggressor.piece_type, 1)
                
                # MVV-LVA score = victim value - aggressor value/10
                # This prioritizes capturing high-value pieces with low-value attackers
                score += victim_value - aggressor_value * 10
        
        # Bonus for promotions
        if move.promotion:
            score += piece_values.get(move.promotion, 0) * 90
        
        # Bonus for attacking center in the opening
        if board.fullmove_number <= 10:
            center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
            if move.to_square in center_squares:
                score += 30
        
        # Bonus for castling
        if board.is_castling(move):
            score += 50
        
        # Bonus for checks
        board.push(move)
        if board.is_check():
            score += 20
        board.pop()
        
        return score
    
    def _iterative_deepening(self, board, max_depth, time_limit):
        """
        Perform iterative deepening search up to max_depth with a time limit.
        
        Args:
            board: A chess.Board object
            max_depth: Maximum search depth
            time_limit: Time limit in seconds
            
        Returns:
            chess.Move: The best move found within the time limit
        """
        start_time = time.time()
        best_move = None
        last_score = 0
        
        # Start with depth 1 and increase gradually
        for depth in range(1, max_depth + 1):
            alpha = float('-inf')
            beta = float('inf')
            best_score = float('-inf')
            current_best_move = None
            
            legal_moves = list(board.legal_moves)
            
            # Use the best move from previous iteration first (if available)
            if best_move in legal_moves:
                legal_moves.remove(best_move)
                legal_moves.insert(0, best_move)
            else:
                legal_moves.sort(key=lambda move: self._move_score(board, move), reverse=True)
            
            for move in legal_moves:
                board.push(move)
                
                # Use negamax with alpha-beta pruning
                score = -self._alpha_beta(board, depth - 1, -beta, -alpha)
                
                board.pop()
                
                if score > best_score:
                    best_score = score
                    current_best_move = move
                
                alpha = max(alpha, score)
                
                # Check if we're out of time
                if time.time() - start_time > time_limit:
                    print(f"Reached time limit at depth {depth}")
                    return best_move if best_move is not None else current_best_move
            
            # Update best move after completing a full depth
            best_move = current_best_move
            
            # Log progress
            elapsed = time.time() - start_time
            print(f"Depth {depth} completed in {elapsed:.2f}s, best move: {board.san(best_move)}, score: {best_score/100:.2f}")
            
            # Score isn't changing significantly between iterations, we might have reached a stable evaluation
            if abs(best_score - last_score) < 10 and depth >= 3 and elapsed > time_limit * 0.75:
                print(f"Stable evaluation reached, stopping at depth {depth}")
                break
                
            last_score = best_score
            
            # Check if we're out of time before starting the next depth
            if time.time() - start_time > time_limit * 0.8:  # Leave a small buffer
                print(f"Time limit approaching, stopping at depth {depth}")
                break
                
        return best_move
    
    def _alpha_beta(self, board, depth, alpha, beta):
        """
        Alpha-beta pruning implementation with negamax framework.
        
        Args:
            board: A chess.Board object
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            
        Returns:
            float: Evaluation score from the perspective of the current player
        """
        self.nodes_evaluated += 1
        
        # Check for immediate game end
        if board.is_game_over():
            if board.is_checkmate():
                # Return a large negative score if current player is checkmated
                return -9900  # Slightly less than max value to prefer quicker mates
            else:
                # Draw
                return 0
        
        # Check transposition table
        board_hash = self._get_board_hash(board)
        if board_hash in self.tt_table and self.tt_table[board_hash]['depth'] >= depth:
            return self.tt_table[board_hash]['score']
        
        # Quiescence search at leaf nodes to handle horizon effect
        if depth == 0:
            return self._quiescence_search(board, alpha, beta)
        
        # Get legal moves and sort them
        legal_moves = list(board.legal_moves)
        legal_moves.sort(key=lambda move: self._move_score(board, move), reverse=True)
        
        best_score = float('-inf')
        
        for move in legal_moves:
            board.push(move)
            score = -self._alpha_beta(board, depth - 1, -beta, -alpha)
            board.pop()
            
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            
            # Alpha-beta cutoff
            if alpha >= beta:
                break
        
        # Store in transposition table
        self.tt_table[board_hash] = {'depth': depth, 'score': best_score}
        return best_score
        
    def _quiescence_search(self, board, alpha, beta, max_depth=3):
        """
        Quiescence search to handle the horizon effect.
        Continues searching captures to reach a "quiet" position.
        
        Args:
            board: A chess.Board object
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            max_depth: Maximum additional depth for quiescence search
            
        Returns:
            float: Evaluation score from the perspective of the current player
        """
        self.nodes_evaluated += 1
        
        # Stand pat - evaluate current position
        stand_pat = self.evaluate_position(board)
        stand_pat = stand_pat if board.turn == chess.WHITE else -stand_pat
        
        if stand_pat >= beta:
            return beta
        
        alpha = max(alpha, stand_pat)
        
        # If we've gone too deep in quiescence, return evaluation
        if max_depth <= 0:
            return stand_pat
        
        # Look at capture moves only
        captures = [move for move in board.legal_moves if board.is_capture(move)]
        captures.sort(key=lambda move: self._move_score(board, move), reverse=True)
        
        for move in captures:
            board.push(move)
            score = -self._quiescence_search(board, -beta, -alpha, max_depth - 1)
            board.pop()
            
            if score >= beta:
                return beta
            
            alpha = max(alpha, score)
        
        return alpha
    
    def _is_capture(self, board, move):
        """
        Check if a move is a capture for move ordering.
        
        Args:
            board: A chess.Board object
            move: A chess.Move object
            
        Returns:
            bool: True if the move is a capture
        """
        return board.is_capture(move)
    
    def _get_board_hash(self, board):
        """
        Get a hash of the current board position for the transposition table.
        
        Args:
            board: A chess.Board object
            
        Returns:
            str: A string hash of the board position
        """
        return board.fen()


def play_game(engine, initial_position=None, max_moves=100):
    """
    Play a complete game using the chess engine.
    
    Args:
        engine: A ChessEngine instance
        initial_position: Optional starting FEN string
        max_moves: Maximum number of moves before forcing a draw
        
    Returns:
        chess.Board: Final board position
        str: Game result
    """
    if initial_position:
        board = chess.Board(initial_position)
    else:
        board = chess.Board()
    
    move_count = 0
    
    while not board.is_game_over() and move_count < max_moves:
        print(f"\nMove {move_count + 1}, {'White' if board.turn == chess.WHITE else 'Black'} to move")
        print(board)
        
        # Use engine to find best move
        start_time = time.time()
        move = engine.get_best_move(board, time_limit=3.0)  # 3 seconds per move
        elapsed = time.time() - start_time
        
        print(f"Engine chose {board.san(move)} in {elapsed:.2f}s")
        board.push(move)
        move_count += 1
    
    print("\nFinal position:")
    print(board)
    
    # Determine result
    if board.is_checkmate():
        result = "1-0" if board.turn == chess.BLACK else "0-1"
        print(f"Checkmate! {'White' if result == '1-0' else 'Black'} wins.")
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        result = "1/2-1/2"
        print("Draw!")
    elif move_count >= max_moves:
        result = "1/2-1/2"
        print(f"Game drawn after {max_moves} moves.")
    else:
        result = "Unknown"
    
    return board, result


if __name__ == "__main__":
    # Example usage
    model_path = "chess_position_evaluator_5000.pkl"
    stockfish_path = "/usr/games/stockfish"
    
    # Initialize parallel position evaluator with multiple instances
    num_evaluators = 15  # Adjust this based on your GPU capacity
    
    try:
        parallel_evaluator = ParallelPositionEvaluator(
            model_path=model_path,
            stockfish_path=stockfish_path,
            num_evaluators=num_evaluators,
            use_gpu=True
        )
        print(f"Using {num_evaluators} parallel evaluators")
    except Exception as e:
        print(f"Failed to initialize parallel evaluator: {e}")
        print("Falling back to single evaluator mode")
        parallel_evaluator = None
        
        # Initialize a single evaluator
        evaluator = ChessPositionEvaluator(stockfish_path=stockfish_path, use_gpu=True)
        try:
            evaluator.load_model(model_path)
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Will use fallback evaluator")
            evaluator = None
    
    # Create chess engine
    engine = ChessEngine(
        position_evaluator=parallel_evaluator if parallel_evaluator else evaluator, 
        stockfish_path=stockfish_path,
        search_depth=4,
        num_parallel_evaluators=num_evaluators
    )
    
    # Example - find best move in current position
    board = chess.Board()
    print("Initial position:")
    print(board)
    
    best_move = engine.get_best_move(board, time_limit=5.0)
    print(f"Best move: {board.san(best_move)}")
    
    # Try with a more interesting position
    fen = "r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
    board = chess.Board(fen)
    print("\nInteresting position (Italian Game):")
    print(board)
    
    best_move = engine.get_best_move(board, time_limit=5.0)
    print(f"Best move: {board.san(best_move)}")
    
    # Clean up resources
    if parallel_evaluator:
        parallel_evaluator.shutdown()
    
    # Uncomment to play a complete game
    # play_game(engine)