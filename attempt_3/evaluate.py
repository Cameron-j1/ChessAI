import chess
import chess.engine
import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Optional
from chess_model_class import ChessModel
from other_functions import board_to_matrix
import random
import time

# Global flag to control verbosity of output
VERBOSE = False

class ChessEvaluator:
    def __init__(self, model_path: str, mapping_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load move mappings
        with open(mapping_path, "rb") as f:
            self.move_to_int = pickle.load(f)
        self.int_to_move = {v: k for k, v in self.move_to_int.items()}
        
        # Load model
        self.model = ChessModel(num_classes=len(self.move_to_int))
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

    def prepare_input(self, board: chess.Board):
        matrix = board_to_matrix(board)
        X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
        return X_tensor

    def checkmate_search(self, board: chess.Board, depth: int = 2, max_depth: int = 5) -> Optional[str]:
        """
        Search for a checkmate sequence with depth extension for checks.
        Base depth is 2 moves, but will extend up to max_depth if checks are found.
        Returns the first move of a checkmate sequence if found, otherwise None.
        """
        # Check immediate checkmates first
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move.uci()
            board.pop()

        def search_position(current_depth: int, alpha_move: chess.Move = None) -> bool:
            if board.is_checkmate():
                return True
            if board.is_stalemate() or board.is_insufficient_material():
                return False
            if current_depth <= 0:
                return False

            moves = list(board.legal_moves)
            # If it's our turn and we're out of moves at a non-zero depth, it's not a win
            if len(moves) == 0:
                return False

            # If it's opponent's turn, they need to have no moves that prevent checkmate
            if board.turn != board.turn_at_start:
                for move in moves:
                    board.push(move)
                    # For opponent moves, extend depth by 1 if it's a check and we haven't hit max_depth
                    new_depth = current_depth - 1
                    if board.is_check() and current_depth + 1 <= max_depth:
                        new_depth += 1
                    if not search_position(new_depth):
                        board.pop()
                        return False
                    board.pop()
                return True
            # If it's our turn, we need at least one move that leads to checkmate
            else:
                for move in moves:
                    board.push(move)
                    # For our moves, extend depth by 1 if it's a check and we haven't hit max_depth
                    new_depth = current_depth - 1
                    if board.is_check() and current_depth + 1 <= max_depth:
                        new_depth += 1
                    if search_position(new_depth):
                        board.pop()
                        if alpha_move is None:  # If this is the first move of the sequence
                            return move
                        return True
                    board.pop()
                return False

        # Start the search from the current position
        result = search_position(depth)
        if isinstance(result, chess.Move):
            return result.uci()
        return None

    def get_piece_value(self, piece: Optional[chess.Piece]) -> int:
        """
        Get the standard point value of a chess piece.
        Returns 0 if piece is None.
        """
        if not piece:
            return 0
            
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King can't really be "traded"
        }
        
        return piece_values.get(piece.piece_type, 0)


    def is_bad_trade(self, our_piece: chess.Piece, captured_piece: chess.Piece, attackers: set, board: chess.Board) -> bool:
        """
        Determine if losing our_piece to recapture would result in a bad trade.
        This now considers multiple attackers and the likely sequence of exchanges.
        Returns True if we would lose more material than we gain.
        """
        # Standard piece values
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King can't really be "traded"
        }
        
        our_piece_value = piece_values.get(our_piece.piece_type, 0)
        captured_piece_value = piece_values.get(captured_piece.piece_type, 0)
        
        # If we captured a piece of value, evaluate the likely exchange sequence
        if captured_piece_value > 0:
            # Get all attacker values, sorted from lowest to highest
            # (opponent will likely attack with cheapest piece first)
            attacker_values = []
            for attacker_square in attackers:
                attacker_piece = board.piece_at(attacker_square)
                if attacker_piece:
                    attacker_value = piece_values.get(attacker_piece.piece_type, 0)
                    attacker_values.append(attacker_value)
            
            attacker_values.sort()  # Cheapest attackers first
            
            if attacker_values:
                # Simple evaluation: if cheapest attacker is worth less than our net gain, it's bad
                cheapest_attacker = attacker_values[0]
                net_material_after_recapture = captured_piece_value - our_piece_value
                
                # If we gain material overall, it might be worth it
                # But if the cheapest attacker is much cheaper than our piece, it's likely bad
                if our_piece_value > captured_piece_value + cheapest_attacker:
                    return True  # We lose too much material in the exchange
                
                # Special case: if there are multiple attackers, be more cautious
                if len(attacker_values) > 1:
                    # If attacked twice or more, and our piece is valuable, be cautious
                    if our_piece_value >= 5 and captured_piece_value < 3:  # Rook+ capturing Knight- or less
                        return True
            
            return False  # Trade seems acceptable
        
        return True  # If captured piece has no value, any loss is bad


    def is_piece_undefended_and_attacked(self, board: chess.Board, square: chess.Square, piece_color: chess.Color) -> Tuple[bool, Optional[chess.Piece], set]:
        """
        Check if a piece of the specified color on a given square is vulnerable to attack.
        A piece is vulnerable if:
        1. It's attacked and completely undefended (always vulnerable), OR
        2. It's attacked more times than it's defended (overloaded), AND
        there exists at least one attacker of equal or lesser value than the piece
        
        Returns (is_vulnerable, piece, attackers) where:
        - is_vulnerable: True if the piece can be won through exchanges
        - piece: The piece object if one exists, None otherwise
        - attackers: Set of squares containing attacking pieces
        """
        piece = board.piece_at(square)
        
        # Only check pieces of the specified color
        if not piece or piece.color != piece_color:
            return False, None, set()
        
        # Get all enemy attackers of the square
        enemy_color = not piece_color
        attackers = board.attackers(enemy_color, square)
        
        if not attackers:  # If no attackers, piece is safe
            return False, piece, set()
        
        # Get all friendly defenders of the square
        defenders = board.attackers(piece_color, square)
        
        # Calculate the exchange balance
        num_attackers = len(attackers)
        num_defenders = len(defenders)
        
        # Case 1: Piece is completely undefended - always vulnerable
        if num_defenders == 0:
            is_vulnerable = True
            if VERBOSE:
                piece_color_str = "White" if piece.color else "Black"
                print(f"Debug - Square: {chess.square_name(square)}, Piece: {piece_color_str} {piece.symbol()}")
                print(f"Debug - Completely undefended piece under attack")
                print(f"Debug - Attackers ({num_attackers}): {[chess.square_name(sq) for sq in attackers]}")
                print(f"Debug - Actually vulnerable: {is_vulnerable}")
            return is_vulnerable, piece, attackers
        
        # Case 2: Check if attackers outnumber defenders (overloaded)
        if num_attackers <= num_defenders:
            return False, piece, attackers  # Not overloaded
        
        # Case 3: Overloaded - check if any attacker would actually want to capture
        piece_value = self.get_piece_value(piece)
        
        # Find the minimum value among attackers
        min_attacker_value = float('inf')
        for attacker_square in attackers:
            attacker_piece = board.piece_at(attacker_square)
            if attacker_piece:
                attacker_value = self.get_piece_value(attacker_piece)
                min_attacker_value = min(min_attacker_value, attacker_value)
        
        # Piece is vulnerable if there's an attacker willing to trade favorably
        # (attacker value <= piece value, or equal values are acceptable trades)
        is_vulnerable = min_attacker_value <= piece_value
        
        # For debugging
        if VERBOSE:
            piece_color_str = "White" if piece.color else "Black"
            print(f"Debug - Square: {chess.square_name(square)}, Piece: {piece_color_str} {piece.symbol()}")
            print(f"Debug - Attackers ({num_attackers}): {[chess.square_name(sq) for sq in attackers]}")
            print(f"Debug - Defenders ({num_defenders}): {[chess.square_name(sq) for sq in defenders]}")
            print(f"Debug - Attack/Defense imbalance: {num_attackers} vs {num_defenders}")
            print(f"Debug - Piece value: {piece_value}, Min attacker value: {min_attacker_value}")
            print(f"Debug - Actually vulnerable: {is_vulnerable}")
        
        return is_vulnerable, piece, attackers


    def has_undefended_pieces_after_move(self, board: chess.Board, move_uci: str, candidate_num: int, move_num: int, is_white: bool) -> bool:
        """
        Checks if making a move would leave any of our pieces undefended and under attack.
        This includes checking if a capturing piece becomes vulnerable after the capture.
        Returns True if there are vulnerable pieces after the move.
        """
        # Create a copy of the board to simulate the move
        board_copy = board.copy()
        
        # Determine our color BEFORE making the move
        our_color = board.turn
        
        # Get the move object and check if it's a capture
        move = chess.Move.from_uci(move_uci)
        is_capture = board.is_capture(move)
        captured_piece = board.piece_at(move.to_square) if is_capture else None
        
        # Make the move
        board_copy.push(move)
        
        # Debug info about the move being checked
        if VERBOSE:
            print(f"\nChecking move {move_uci} for hanging pieces:")
            print(f"Our color: {'White' if our_color else 'Black'}")
            print(f"Is capture: {is_capture}")
            if captured_piece:
                print(f"Captured piece: {captured_piece.symbol()}")
        
        # Calculate total material we could lose from hanging pieces
        total_material_at_risk = 0
        hanging_pieces_info = []
        
        # Check all our pieces for being vulnerable (including the piece that just moved)
        for square in chess.SQUARES:
            piece = board_copy.piece_at(square)
            if piece and piece.color == our_color:  # Only check our pieces
                result, piece_obj, attackers = self.is_piece_undefended_and_attacked(board_copy, square, our_color)
                if result:
                    piece_value = self.get_piece_value(piece_obj)
                    total_material_at_risk += piece_value
                    square_name = chess.square_name(square)
                    attacker_squares = [chess.square_name(attacker) for attacker in attackers]
                    defenders = board_copy.attackers(our_color, square)
                    defender_count = len(defenders)
                    attacker_count = len(attackers)
                    
                    hanging_pieces_info.append({
                        'piece': piece_obj,
                        'square_name': square_name,
                        'value': piece_value,
                        'attackers': attacker_squares,
                        'attacker_count': attacker_count,
                        'defender_count': defender_count
                    })
        
        # If we have hanging pieces, evaluate if the trade is worth it
        if hanging_pieces_info:
            material_gained = self.get_piece_value(captured_piece) if captured_piece else 0
            net_material_balance = material_gained - total_material_at_risk
            
            # Only reject the move if we lose material overall
            if net_material_balance < 0:
                if VERBOSE:
                    move_str = f"{move_num}." if is_white else f"{move_num}..."
                    
                    if is_capture and captured_piece:
                        print(f"Move {move_str} Candidate #{candidate_num}: Skipping move {move_uci} as capturing {captured_piece.symbol()} (+{material_gained}) but losing {total_material_at_risk} points in hanging pieces results in net loss of {abs(net_material_balance)} points")
                    elif is_capture:
                        # Handle en passant or other special captures where captured_piece might be None
                        print(f"Move {move_str} Candidate #{candidate_num}: Skipping move {move_uci} as capture (+{material_gained}) but losing {total_material_at_risk} points in hanging pieces results in net loss of {abs(net_material_balance)} points")
                    else:
                        print(f"Move {move_str} Candidate #{candidate_num}: Skipping move {move_uci} as it leaves {total_material_at_risk} points of material hanging")
                    
                    # Print details of each hanging piece
                    for info in hanging_pieces_info:
                        print(f"  - {info['piece'].symbol()} on {info['square_name']} ({info['attacker_count']} attackers vs {info['defender_count']} defenders): {', '.join(info['attackers'])}")
                
                return True
            elif VERBOSE:
                # Material balance is favorable or neutral, allow the move
                captured_desc = captured_piece.symbol() if captured_piece else "piece"
                print(f"Move analysis: Capturing {captured_desc} (+{material_gained}) vs losing {total_material_at_risk} points = net gain of {net_material_balance} points. Move allowed.")
                return False
        
        return False

    def validate_model_move(self, board: chess.Board, move_uci: str, candidate_num: int, move_num: int, is_white: bool) -> bool:
        """
        Validates if a move is safe by checking:
        1. If it allows opponent to checkmate immediately
        2. If it leaves any pieces undefended and under attack
        Returns True if the move is safe, False otherwise.
        """
        # First check if move allows immediate checkmate
        board_copy = board.copy()
        move = chess.Move.from_uci(move_uci)
        board_copy.push(move)
        
        # Check for immediate checkmate
        for opponent_move in board_copy.legal_moves:
            board_copy.push(opponent_move)
            if board_copy.is_checkmate():
                if VERBOSE:
                    move_str = f"{move_num}." if is_white else f"{move_num}..."
                    print(f"Move {move_str} Candidate #{candidate_num}: Skipping move {move_uci} as it allows immediate checkmate by {opponent_move.uci()}")
                board_copy.pop()
                return False
            board_copy.pop()
            
        # Then check for undefended pieces
        if self.has_undefended_pieces_after_move(board, move_uci, candidate_num, move_num, is_white):
            return False
            
        return True

    def get_model_move(self, board: chess.Board, move_num: int, is_white: bool) -> Optional[str]:
        # First check for checkmate sequences
        board.turn_at_start = board.turn  # Add this attribute for the search
        checkmate_move = self.checkmate_search(board)
        if checkmate_move:
            if VERBOSE:
                move_str = f"{move_num}." if is_white else f"{move_num}..."
                print(f"Move {move_str} Found checkmate sequence! Playing first move of sequence.")
            return checkmate_move
            
        # If no checkmate found, use the model
        X_tensor = self.prepare_input(board).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
        
        logits = logits.squeeze(0)
        probabilities = torch.softmax(logits, dim=0).cpu().numpy()
        legal_moves = list(board.legal_moves)
        legal_moves_uci = [move.uci() for move in legal_moves]
        sorted_indices = np.argsort(probabilities)[::-1]
        
        # Try moves in order of model preference until finding a safe one
        for candidate_num, move_index in enumerate(sorted_indices, 1):
            move = self.int_to_move[move_index]
            if move in legal_moves_uci:
                # Validate the move doesn't allow immediate checkmate
                if self.validate_model_move(board, move, candidate_num, move_num, is_white):
                    if VERBOSE:
                        move_str = f"{move_num}." if is_white else f"{move_num}..."
                        print(f"Move {move_str} Selected candidate #{candidate_num}: {move}")
                    return move
        
        # If we get here, either no legal moves or all moves allow checkmate
        if legal_moves:
            if VERBOSE:
                move_str = f"{move_num}." if is_white else f"{move_num}..."
                print(f"Move {move_str} Warning: All candidates were rejected. Falling back to first legal move.")
            # If all moves are bad, just return the first legal move
            return legal_moves[0].uci()
        return None

def play_game(white_player, black_player, num_games: int = 1) -> Tuple[int, int, int]:
    wins_white = 0
    wins_black = 0
    draws = 0
    
    for game_num in range(num_games):
        board = chess.Board()
        move_count = 0
        game_moves = []  # Store moves for PGN
        
        print(f"\nPlaying game {game_num + 1}/{num_games}")
        
        while not board.is_game_over():
            current_move_num = (move_count // 2) + 1
            if board.turn:  # White's turn
                if isinstance(white_player, str) and white_player == "random":
                    move = random.choice(list(board.legal_moves))
                    board.push(move)
                    game_moves.append(move)
                elif isinstance(white_player, chess.engine.SimpleEngine):
                    result = white_player.play(board, chess.engine.Limit(time=0.1))
                    board.push(result.move)
                    game_moves.append(result.move)
                else:  # ChessEvaluator instance
                    move = white_player.get_model_move(board, current_move_num, True)
                    if move:
                        move_obj = board.push_uci(move)
                        game_moves.append(move_obj)
                    else:
                        if VERBOSE:
                            print("White couldn't find a valid move!")
                        break
            else:  # Black's turn
                if isinstance(black_player, str) and black_player == "random":
                    move = random.choice(list(board.legal_moves))
                    board.push(move)
                    game_moves.append(move)
                elif isinstance(black_player, chess.engine.SimpleEngine):
                    result = black_player.play(board, chess.engine.Limit(time=0.1))
                    board.push(result.move)
                    game_moves.append(result.move)
                else:  # ChessEvaluator instance
                    move = black_player.get_model_move(board, current_move_num, False)
                    if move:
                        move_obj = board.push_uci(move)
                        game_moves.append(move_obj)
                    else:
                        if VERBOSE:
                            print("Black couldn't find a valid move!")
                        break
            
            move_count += 1
        
        # Create a new board for PGN generation
        pgn_board = chess.Board()
        print("\nGame PGN:", end=" ")
        for i, move in enumerate(game_moves):
            if i % 2 == 0:
                print(f"{i//2 + 1}. {pgn_board.san(move)}", end=" ")
            else:
                print(f"{pgn_board.san(move)}", end=" ")
            pgn_board.push(move)
        print()
        
        result = board.outcome()
        if result:
            if result.winner == chess.WHITE:
                wins_white += 1
                print(f"Game {game_num + 1}: White wins")
            elif result.winner == chess.BLACK:
                wins_black += 1
                print(f"Game {game_num + 1}: Black wins")
            else:
                draws += 1
                print(f"Game {game_num + 1}: Draw")
        else:
            draws += 1
            print(f"Game {game_num + 1}: Draw (no valid moves)")
            
    return wins_white, wins_black, draws

def main():
    # Configuration variables - modify these as needed
    # mode = 'random'  # Choose from: 'random', 'model', or 'stockfish'
    # mode = 'model'  
    mode = 'stockfish'
    # model_path = 'models/model_architecture_v2_epochs20.pth'  # Path to your main model
    model_path = 'models/trainv2_model_architecture_v2_epochs20.pth'  # Path to your main model
    opponent_model_path = None  # Path to opponent model (for mode='model')
    stockfish_path = '/usr/games/stockfish'  # Path to stockfish executable (for mode='stockfish')
    stockfish_elo = 1350  # Stockfish ELO rating
    num_games = 100  # Number of games to play
    time_limit = 0.001  # Time limit per move for Stockfish (in seconds)
    
    # Initialize main model
    main_model = ChessEvaluator(model_path, "models/move_to_int_architecture_v2")
    
    if mode == 'random':
        # Play against random moves
        if VERBOSE:
            print("\nEvaluating against random moves...")
        wins_white, wins_black, draws = play_game(main_model, "random", num_games)
        
    elif mode == 'model':
        if not opponent_model_path:
            raise ValueError("opponent_model_path must be set for model mode")
        
        # Play against another model
        if VERBOSE:
            print("\nEvaluating against another model...")
        opponent_model = ChessEvaluator(opponent_model_path, "models/move_to_int")
        wins_white, wins_black, draws = play_game(main_model, opponent_model, num_games)
        
    elif mode == 'stockfish':
        if not stockfish_path:
            raise ValueError("stockfish_path must be set for stockfish mode")
        
        # Play against Stockfish
        if VERBOSE:
            print(f"\nEvaluating against Stockfish (ELO: {stockfish_elo})...")
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"UCI_LimitStrength": stockfish_elo})  # Approximate ELO to skill level
        
        wins_white, wins_black, draws = play_game(main_model, engine, num_games)
        engine.quit()
    
    else:
        raise ValueError("Invalid mode. Choose from: 'random', 'model', or 'stockfish'")
    
    # Print results (these always show regardless of VERBOSE setting)
    print("\nEvaluation Results:")
    print(f"Total games: {num_games}")
    print(f"Main model wins (as White): {wins_white}")
    print(f"Opponent wins: {wins_black}")
    print(f"Draws: {draws}")
    print(f"Win rate: {(wins_white / num_games) * 100:.2f}%")
    print(f"Draw rate: {(draws / num_games) * 100:.2f}%")

if __name__ == "__main__":
    main() 