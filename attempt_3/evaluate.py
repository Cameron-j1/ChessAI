import chess
import chess.engine
import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Optional, List
# Import models conditionally based on model type
from chess_model_class import ChessModel, ChessModelSpatial as ChessModelSpatialBase
from chess_model_class_attention import ChessModelAttention
from other_functions import board_to_matrix, decode_move_spatial
import random
import time

# Global flag to control verbosity of output
VERBOSE = False
# VERBOSE = True

class ChessEvaluator:
    def __init__(self, model_path: str, mapping_path: str = None, model_type: str = "categorical"):
        """
        Initialize ChessEvaluator with support for categorical, spatial, and attention models.
        
        Args:
            model_path: Path to the model file
            mapping_path: Path to move mapping file (only needed for categorical models)
            model_type: Either "categorical", "spatial", or "attention"
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        
        if model_type == "categorical":
            if mapping_path is None:
                raise ValueError("mapping_path is required for categorical models")
            
            # Load move mappings for categorical model
            with open(mapping_path, "rb") as f:
                self.move_to_int = pickle.load(f)
            self.int_to_move = {v: k for k, v in self.move_to_int.items()}
            
            # Load categorical model
            self.model = ChessModel(num_classes=len(self.move_to_int))
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
        elif model_type == "spatial":
            # Load spatial model (no move mappings needed)
            self.model = ChessModelSpatialBase()
            # Load checkpoint and extract model state dict
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # If it's a training checkpoint, extract just the model state dict
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # If it's already a state dict, load it directly
                self.model.load_state_dict(checkpoint)
            self.move_to_int = None
            self.int_to_move = None
            
        elif model_type == "attention":
            # Load attention model (no move mappings needed)
            self.model = ChessModelAttention()
            # Load checkpoint and extract model state dict
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # If it's a training checkpoint, extract just the model state dict
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # If it's already a state dict, load it directly
                self.model.load_state_dict(checkpoint)
            self.move_to_int = None
            self.int_to_move = None
            
        else:
            raise ValueError("model_type must be either 'categorical', 'spatial', or 'attention'")
        
        self.model.to(self.device)
        self.model.eval()

    def prepare_input(self, board: chess.Board):
        matrix = board_to_matrix(board)
        X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
        return X_tensor

    def checkmate_search(self, board: chess.Board, depth: int = 3, max_depth: int = 8) -> Optional[str]:
        """
        Search for a checkmate sequence with depth extension for checks.
        Base depth is 3 moves, but will extend up to max_depth if checks continue.
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

    def get_xray_attackers_defenders(self, board: chess.Board, square: chess.Square, piece_color: chess.Color) -> Tuple[set, set]:
        """
        Get both direct and X-ray attackers/defenders for a square.
        Returns (attackers, defenders) where each is a set of squares.
        Includes pieces that can attack/defend through other pieces in their line of sight.
        """
        # Get initial attackers and defenders
        direct_attackers = board.attackers(not piece_color, square)
        direct_defenders = board.attackers(piece_color, square)
        
        # Initialize sets for all attackers/defenders (including X-rays)
        all_attackers = set(direct_attackers)
        all_defenders = set(direct_defenders)
        
        # Helper function to check for X-ray pieces along a ray
        def check_ray(start_square: chess.Square, direction: int, friendly: bool) -> Optional[chess.Square]:
            current = start_square
            pieces_in_ray = 0
            last_piece_square = None
            
            while True:
                current += direction
                if not (0 <= current < 64):  # Off the board
                    break
                    
                # Check if we're still on the same rank/file/diagonal
                if (abs(chess.square_file(current) - chess.square_file(current - direction)) > 1 or
                    abs(chess.square_rank(current) - chess.square_rank(current - direction)) > 1):
                    break
                
                piece = board.piece_at(current)
                if piece is not None:
                    pieces_in_ray += 1
                    last_piece_square = current
                    
                    # If we've found more than one piece, check if it's a valid X-ray attacker/defender
                    if pieces_in_ray > 1:
                        if piece.color == (piece_color if friendly else not piece_color):
                            piece_type = piece.piece_type
                            # Check if piece can move in this direction
                            if ((piece_type == chess.ROOK and direction in (-8, -1, 1, 8)) or
                                (piece_type == chess.BISHOP and direction in (-9, -7, 7, 9)) or
                                (piece_type == chess.QUEEN)):
                                return last_piece_square
                        break
                    
            return None
        
        # Directions for all possible rays
        directions = {
            chess.ROOK: (-8, -1, 1, 8),  # Up, left, right, down
            chess.BISHOP: (-9, -7, 7, 9),  # Diagonals
        }
        
        # Check for X-ray attackers and defenders
        for piece_type, dirs in directions.items():
            for d in dirs:
                # Check for X-ray attackers
                xray_square = check_ray(square, d, False)
                if xray_square is not None:
                    all_attackers.add(xray_square)
                
                # Check for X-ray defenders
                xray_square = check_ray(square, d, True)
                if xray_square is not None:
                    all_defenders.add(xray_square)
        
        return all_attackers, all_defenders

    def is_piece_undefended_and_attacked(self, board: chess.Board, square: chess.Square, piece_color: chess.Color) -> Tuple[bool, Optional[chess.Piece], set]:
        """
        Check if a piece of the specified color on a given square is vulnerable to attack.
        X-ray attackers and defenders are only considered in overloaded situations.
        Returns (is_vulnerable, piece, attackers).
        """
        piece = board.piece_at(square)
        
        # Only check pieces of the specified color
        if not piece or piece.color != piece_color:
            return False, None, set()
        
        # Get direct attackers and defenders first (convert SquareSet to regular set)
        direct_attackers = set(board.attackers(not piece_color, square))
        direct_defenders = set(board.attackers(piece_color, square))
        
        # Get all attackers and defenders, including X-rays
        all_attackers, all_defenders = self.get_xray_attackers_defenders(board, square, piece_color)
        
        # Separate X-ray attackers and defenders
        xray_attackers = all_attackers - direct_attackers
        xray_defenders = all_defenders - direct_defenders
        
        # Start with direct attackers and defenders
        attackers = direct_attackers
        defenders = direct_defenders
        
        if not attackers:  # If no direct attackers, piece is safe
            return False, piece, set()
        
        # Get piece values
        piece_value = self.get_piece_value(piece)
        
        # Calculate initial counts
        num_direct_attackers = len(direct_attackers)
        num_direct_defenders = len(direct_defenders)
        
        # Find the minimum value among direct attackers
        min_attacker_value = float('inf')
        for attacker_square in direct_attackers:
            attacker_piece = board.piece_at(attacker_square)
            if attacker_piece:
                attacker_value = self.get_piece_value(attacker_piece)
                min_attacker_value = min(min_attacker_value, attacker_value)
        
        # Find the minimum value among direct defenders
        min_defender_value = float('inf')
        for defender_square in direct_defenders:
            defender_piece = board.piece_at(defender_square)
            if defender_piece:
                defender_value = self.get_piece_value(defender_piece)
                min_defender_value = min(min_defender_value, defender_value)
        
        # Case 1: Piece is completely undefended (no direct defenders)
        if num_direct_defenders == 0:
            is_vulnerable = True
            # In this case, we don't consider X-ray defenders as they can't help immediately
        
        # Case 2: Piece has defenders but might be overloaded
        elif num_direct_attackers > num_direct_defenders:
            # In overloaded situations, consider X-ray attackers and defenders
            attackers = all_attackers  # Include X-ray attackers
            defenders = all_defenders  # Include X-ray defenders
            num_total_attackers = len(attackers)
            num_total_defenders = len(defenders)
            
            # Recalculate minimum values including X-rays
            for attacker_square in xray_attackers:
                attacker_piece = board.piece_at(attacker_square)
                if attacker_piece:
                    attacker_value = self.get_piece_value(attacker_piece)
                    min_attacker_value = min(min_attacker_value, attacker_value)
            
            for defender_square in xray_defenders:
                defender_piece = board.piece_at(defender_square)
                if defender_piece:
                    defender_value = self.get_piece_value(defender_piece)
                    min_defender_value = min(min_defender_value, defender_value)
            
            # Piece is vulnerable if:
            # 1. Still overloaded even with X-ray defenders AND
            # 2. The cheapest attacker is worth less than or equal to the piece AND
            # 3. The cheapest attacker is worth less than or equal to the cheapest defender
            is_vulnerable = (num_total_attackers > num_total_defenders and
                           min_attacker_value <= piece_value and 
                           min_attacker_value <= min_defender_value)
        
        # Case 3: Single attacker (don't consider X-rays)
        elif num_direct_attackers == 1:
            is_vulnerable = min_attacker_value < piece_value
        
        # Case 4: Multiple attackers but not overloaded (don't consider X-rays)
        else:
            is_vulnerable = False
        
        # For debugging
        if VERBOSE:
            piece_color_str = "White" if piece.color else "Black"
            print(f"Debug - Square: {chess.square_name(square)}, Piece: {piece_color_str} {piece.symbol()}")
            print(f"Debug - Direct Attackers ({num_direct_attackers}): {[chess.square_name(sq) for sq in direct_attackers]}")
            if xray_attackers:
                print(f"Debug - X-ray Attackers ({len(xray_attackers)}): {[chess.square_name(sq) for sq in xray_attackers]}")
            if num_direct_defenders > 0:
                print(f"Debug - Direct Defenders ({num_direct_defenders}): {[chess.square_name(sq) for sq in direct_defenders]}")
                if xray_defenders:
                    print(f"Debug - X-ray Defenders ({len(xray_defenders)}): {[chess.square_name(sq) for sq in xray_defenders]}")
                print(f"Debug - Attack/Defense imbalance: {len(attackers)} vs {len(defenders)} (including X-rays where relevant)")
                print(f"Debug - Piece value: {piece_value}, Min attacker value: {min_attacker_value}, Min defender value: {min_defender_value}")
            else:
                print(f"Debug - Completely undefended piece under attack")
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
        captured_piece = board_copy.piece_at(move.to_square) if is_capture else None
        
        # Make the move
        board_copy.push(move)
        
        # Debug info about the move being checked
        if VERBOSE:
            print(f"\nChecking move {move_uci} for hanging pieces:")
            print(f"Our color: {'White' if our_color else 'Black'}")
            print(f"Is capture: {is_capture}")
            if captured_piece:
                print(f"Captured piece: {captured_piece.symbol()}")
        
        # Find all hanging pieces and their values
        hanging_pieces_info = []
        
        # Check all our pieces for being vulnerable (including the piece that just moved)
        for square in chess.SQUARES:
            piece = board_copy.piece_at(square)
            if piece and piece.color == our_color:  # Only check our pieces
                result, piece_obj, attackers = self.is_piece_undefended_and_attacked(board_copy, square, our_color)
                if result:
                    piece_value = self.get_piece_value(piece_obj)
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
            
            # The opponent can only capture one piece per move, so we only risk the most valuable hanging piece
            # (assuming the opponent plays optimally and takes the most valuable piece first)
            most_valuable_hanging_piece = max(hanging_pieces_info, key=lambda x: x['value'])['value']
            total_material_at_risk = sum(info['value'] for info in hanging_pieces_info)  # For display purposes
            
            # Calculate net material balance considering only the most valuable piece at risk
            net_material_balance = material_gained - most_valuable_hanging_piece
            
            # Only reject the move if we lose material in the immediate exchange
            if net_material_balance < 0:
                if VERBOSE:
                    move_str = f"{move_num}." if is_white else f"{move_num}..."
                    
                    if is_capture and captured_piece:
                        print(f"Move {move_str} Candidate #{candidate_num}: Skipping move {move_uci} as capturing {captured_piece.symbol()} (+{material_gained}) but losing most valuable hanging piece ({most_valuable_hanging_piece} points) results in net loss of {abs(net_material_balance)} points")
                    elif is_capture:
                        # Handle en passant or other special captures where captured_piece might be None
                        print(f"Move {move_str} Candidate #{candidate_num}: Skipping move {move_uci} as capture (+{material_gained}) but losing most valuable hanging piece ({most_valuable_hanging_piece} points) results in net loss of {abs(net_material_balance)} points")
                    else:
                        print(f"Move {move_str} Candidate #{candidate_num}: Skipping move {move_uci} as it leaves {most_valuable_hanging_piece} points of material hanging (most valuable piece)")
                    
                    # Print details of each hanging piece
                    print(f"  Total material at risk: {total_material_at_risk} points, but opponent can only take {most_valuable_hanging_piece} points immediately:")
                    for info in hanging_pieces_info:
                        marker = " ← MOST VALUABLE" if info['value'] == most_valuable_hanging_piece else ""
                        print(f"  - {info['piece'].symbol()} on {info['square_name']} ({info['attacker_count']} attackers vs {info['defender_count']} defenders): {', '.join(info['attackers'])}{marker}")
                
                return True
            elif VERBOSE:
                # Material balance is favorable or neutral, allow the move
                captured_desc = captured_piece.symbol() if captured_piece else "piece"
                print(f"Move analysis: Capturing {captured_desc} (+{material_gained}) vs losing most valuable hanging piece ({most_valuable_hanging_piece} points) = net gain of {net_material_balance} points. Move allowed.")
                if len(hanging_pieces_info) > 1:
                    print(f"  Note: {len(hanging_pieces_info)} pieces hanging (total {total_material_at_risk} points), but opponent can only capture one per move.")
                return False
        
        return False

    def check_for_checkmate_threat(self, board: chess.Board, move_uci: str, candidate_num: int, move_num: int, is_white: bool, max_depth: int = 3, max_check_extensions: int = 8) -> bool:
        """
        Check if a move allows opponent to force checkmate within max_depth moves.
        Extends search up to max_check_extensions additional moves if checks are found.
        
        Args:
            board: Current board position
            move_uci: Candidate move to evaluate
            candidate_num: Number of the candidate move (for logging)
            move_num: Current move number in the game
            is_white: Whether it's white's turn
            max_depth: Maximum depth to search (default 3)
            max_check_extensions: Maximum number of additional moves to search when checks are found (default 3)
            
        Returns:
            bool: True if move allows forced checkmate, False otherwise
        """
        board_copy = board.copy()
        move = chess.Move.from_uci(move_uci)
        board_copy.push(move)
        
        def search_for_mate(position: chess.Board, depth: int, extensions_left: int) -> Tuple[bool, Optional[List[chess.Move]]]:
            """
            Recursive helper function to search for forced checkmate.
            Returns (found_mate, mating_sequence)
            """
            if position.is_checkmate():
                return True, []
                
            if depth <= 0 and not position.is_check():
                return False, None
                
            if depth <= -max_check_extensions:  # Hard limit on extensions
                return False, None
            
            # If it's a check and we have extensions left, we'll search one more ply
            is_checking = position.is_check()
            can_extend = extensions_left > 0 if depth <= 0 else True
            
            legal_moves = list(position.legal_moves)
            if len(legal_moves) == 0:  # Stalemate
                return False, None
                
            if position.turn == board_copy.turn:  # Opponent's turn (looking for mate)
                # Try each move, looking for any that leads to mate
                for move in legal_moves:
                    position.push(move)
                    new_depth = depth - 1
                    new_extensions = extensions_left
                    if is_checking and can_extend and depth <= 0:
                        new_depth = 0  # Keep searching at depth 0
                        new_extensions = extensions_left - 1
                    found_mate, mate_sequence = search_for_mate(position, new_depth, new_extensions)
                    position.pop()
                    
                    if found_mate:
                        if mate_sequence is not None:
                            mate_sequence.insert(0, move)
                        else:
                            mate_sequence = [move]
                        return True, mate_sequence
                return False, None
            else:  # Our turn (defending against mate)
                # Need to prevent mate in all variations
                for move in legal_moves:
                    position.push(move)
                    new_depth = depth - 1
                    new_extensions = extensions_left
                    if is_checking and can_extend and depth <= 0:
                        new_depth = 0  # Keep searching at depth 0
                        new_extensions = extensions_left - 1
                    found_mate, mate_sequence = search_for_mate(position, new_depth, new_extensions)
                    position.pop()
                    
                    if not found_mate:  # Found a defense
                        return False, None
                return True, []  # No defense found
        
        # Start the search
        found_mate, mate_sequence = search_for_mate(board_copy, max_depth, max_check_extensions)
        
        if found_mate and VERBOSE:
            move_str = f"{move_num}." if is_white else f"{move_num}..."
            sequence_str = " -> ".join([m.uci() for m in mate_sequence]) if mate_sequence else "immediate mate"
            print(f"Move {move_str} Candidate #{candidate_num}: Skipping move {move_uci} as it allows forced checkmate sequence: {sequence_str}")
        
        return found_mate

    def get_model_move(self, board: chess.Board, move_num: int, is_white: bool, output_LAN: bool = False) -> Optional[str]:
        # First check for checkmate sequences
        board.turn_at_start = board.turn  # Add this attribute for the search
        
        check_extensions = 8
        
        # Count total pieces on the board
        total_pieces = len(list(board.pieces(chess.PAWN, chess.WHITE))) + len(list(board.pieces(chess.PAWN, chess.BLACK))) + \
                      len(list(board.pieces(chess.KNIGHT, chess.WHITE))) + len(list(board.pieces(chess.KNIGHT, chess.BLACK))) + \
                      len(list(board.pieces(chess.BISHOP, chess.WHITE))) + len(list(board.pieces(chess.BISHOP, chess.BLACK))) + \
                      len(list(board.pieces(chess.ROOK, chess.WHITE))) + len(list(board.pieces(chess.ROOK, chess.BLACK))) + \
                      len(list(board.pieces(chess.QUEEN, chess.WHITE))) + len(list(board.pieces(chess.QUEEN, chess.BLACK))) + \
                      len(list(board.pieces(chess.KING, chess.WHITE))) + len(list(board.pieces(chess.KING, chess.BLACK)))
        
        # Extend search depth in endgame positions
        base_depth = 3
        if total_pieces <= 10:  # Endgame with 10 or fewer pieces
            base_depth = 4  # Base depth + 8 as requested
            check_extensions = 9
            if VERBOSE:
                print(f"Endgame position detected ({total_pieces} pieces). Extending checkmate search depth to {base_depth}.")
        
        checkmate_move = self.checkmate_search(board, depth=base_depth, max_depth=check_extensions)
        if checkmate_move:
            # If it's a promotion move, ensure it promotes to queen
            if len(checkmate_move) == 5:  # Promotion moves are 5 characters long
                checkmate_move = checkmate_move[:4] + 'q'  # Force queen promotion
            if VERBOSE:
                move_str = f"{move_num}." if is_white else f"{move_num}..."
                print(f"Move {move_str} Found checkmate sequence! Playing first move of sequence.")
            if output_LAN:
                move_obj = chess.Move.from_uci(checkmate_move)
                return board.lan(move_obj)
            return checkmate_move
            
        # If no checkmate found, use the model
        X_tensor = self.prepare_input(board).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
        
        legal_moves = list(board.legal_moves)
        legal_moves_uci = [move.uci() for move in legal_moves]
        
        if self.model_type == "categorical":
            # Handle categorical model (original behavior)
            logits = logits.squeeze(0)
            probabilities = torch.softmax(logits, dim=0).cpu().numpy()
            sorted_indices = np.argsort(probabilities)[::-1]
            
            # Try moves in order of model preference until finding a safe one
            rejected_moves = []  # Track rejected moves and their material loss
            for candidate_num, move_index in enumerate(sorted_indices, 1):
                move = self.int_to_move[move_index]
                # If it's a promotion move, ensure it promotes to queen
                if len(move) == 5:  # Promotion moves are 5 characters long
                    move = move[:4] + 'q'  # Force queen promotion
                if move in legal_moves_uci:
                    # Validate the move using existing logic
                    if self._validate_and_check_move(board, move, candidate_num, move_num, is_white, rejected_moves):
                        if output_LAN:
                            move_obj = chess.Move.from_uci(move)
                            return board.lan(move_obj)
                        return move
            
        elif self.model_type in ["spatial", "attention"]:  # Handle both spatial and attention models the same way
            # Handle spatial/attention model
            logits = logits.squeeze(0)  # Shape: (2, 8, 8)
            
            # Apply softmax to convert logits to probabilities
            from_probs = torch.softmax(logits[0].flatten(), dim=0).view(8, 8)
            to_probs = torch.softmax(logits[1].flatten(), dim=0).view(8, 8)
            
            # Stack probabilities
            probabilities = torch.stack([from_probs, to_probs], dim=0).cpu().numpy()
            
            # Get candidate moves from spatial prediction
            candidate_moves = self._get_spatial_candidate_moves(probabilities, legal_moves, top_k=50)
            
            # Try moves in order of spatial model preference
            rejected_moves = []
            for candidate_num, move in enumerate(candidate_moves, 1):
                # If it's a promotion move, ensure it promotes to queen
                if len(move) == 5:  # Promotion moves are 5 characters long
                    move = move[:4] + 'q'  # Force queen promotion
                if self._validate_and_check_move(board, move, candidate_num, move_num, is_white, rejected_moves):
                    if output_LAN:
                        move_obj = chess.Move.from_uci(move)
                        return board.lan(move_obj)
                    return move
        
        # If we get here, all moves were rejected
        if rejected_moves:
            # Sort rejected moves by net material loss (primary) and total hanging value (secondary)
            rejected_moves.sort(key=lambda x: (x[1], x[2]))
            best_bad_move, loss, total_loss, candidate_num = rejected_moves[0]
            # If it's a promotion move, ensure it promotes to queen
            if len(best_bad_move) == 5:  # Promotion moves are 5 characters long
                best_bad_move = best_bad_move[:4] + 'q'  # Force queen promotion
            
            if VERBOSE:
                move_str = f"{move_num}." if is_white else f"{move_num}..."
                print(f"Move {move_str} Warning: All candidates were rejected. Choosing least bad option:")
                print(f"Selected move {best_bad_move} (candidate #{candidate_num}) with net material loss of {loss} points")
                if len(rejected_moves) > 1:
                    print(f"Other options had higher losses ranging from {min(m[1] for m in rejected_moves[1:])} to {max(m[1] for m in rejected_moves[1:])} points")
                else:
                    print("This was the only candidate move available")
            if output_LAN:
                move_obj = chess.Move.from_uci(best_bad_move)
                return board.lan(move_obj)
            return best_bad_move
        
        # If we get here, either no legal moves or all moves allow checkmate
        if legal_moves:
            move = legal_moves[0].uci()
            # If it's a promotion move, ensure it promotes to queen
            if len(move) == 5:  # Promotion moves are 5 characters long
                move = move[:4] + 'q'  # Force queen promotion
            if output_LAN:
                move_obj = chess.Move.from_uci(move)
                return board.lan(move_obj)
            if VERBOSE:
                move_str = f"{move_num}." if is_white else f"{move_num}..."
                print(f"Move {move_str} Warning: No valid moves found (all allow checkmate). Falling back to first legal move.")
            return move
        return None

    def _get_spatial_candidate_moves(self, probabilities, legal_moves, top_k=50):
        """
        Get candidate moves from spatial model predictions, sorted by probability.
        
        Args:
            probabilities: numpy array of shape (2, 8, 8) - from/to probabilities
            legal_moves: list of legal chess.Move objects
            top_k: maximum number of candidate moves to return
            
        Returns:
            list of UCI move strings sorted by probability (highest first)
        """
        from_probs = probabilities[0].flatten()
        to_probs = probabilities[1].flatten()
        
        # Convert legal moves to (from_square, to_square) tuples for quick lookup
        legal_move_tuples = [(move.from_square, move.to_square) for move in legal_moves]
        legal_moves_dict = {(move.from_square, move.to_square): move.uci() for move in legal_moves}
        
        # Calculate combined probabilities for all legal moves
        move_probabilities = []
        for from_square, to_square in legal_move_tuples:
            combined_prob = from_probs[from_square] * to_probs[to_square]
            move_probabilities.append((legal_moves_dict[(from_square, to_square)], combined_prob))
        
        # Sort by probability (highest first) and return top_k moves
        move_probabilities.sort(key=lambda x: x[1], reverse=True)
        return [move for move, prob in move_probabilities[:top_k]]

    def _validate_and_check_move(self, board, move, candidate_num, move_num, is_white, rejected_moves):
        """
        Validate a move and check if it's safe. Used by both categorical and spatial models.
        
        Returns:
            bool: True if move is safe and should be played, False otherwise
        """
        # Create a copy of the board to analyze the position
        board_copy = board.copy()
        move_obj = chess.Move.from_uci(move)
        
        # Get material value of any captured piece
        captured_piece = board_copy.piece_at(move_obj.to_square)
        material_gained = self.get_piece_value(captured_piece) if captured_piece else 0
        
        # Make the move on the copy
        board_copy.push(move_obj)
        
        # Check for stalemate - reject move if it causes stalemate
        if board_copy.is_stalemate():
            if VERBOSE:
                move_str = f"{move_num}." if is_white else f"{move_num}..."
                print(f"Move {move_str} Candidate #{candidate_num}: Skipping move {move} as it causes stalemate")
            return False
        
        # Check for immediate checkmate (this is still an instant rejection)
        checkmate_allowed = False
        for opponent_move in board_copy.legal_moves:
            board_copy.push(opponent_move)
            if board_copy.is_checkmate():
                checkmate_allowed = True
            board_copy.pop()
        
        if checkmate_allowed:
            if VERBOSE:
                move_str = f"{move_num}." if is_white else f"{move_num}..."
                print(f"Move {move_str} Candidate #{candidate_num}: Skipping move {move} as it allows immediate checkmate")
            return False
        
        # Find hanging pieces after the move
        total_hanging_value = 0
        most_valuable_hanging = 0
        
        for square in chess.SQUARES:
            piece = board_copy.piece_at(square)
            if piece and piece.color == board.turn:  # Only check our pieces
                result, piece_obj, attackers = self.is_piece_undefended_and_attacked(board_copy, square, board.turn)
                if result:
                    piece_value = self.get_piece_value(piece_obj)
                    most_valuable_hanging = max(most_valuable_hanging, piece_value)
                    total_hanging_value += piece_value
        
        # Calculate net material change
        net_material_loss = most_valuable_hanging - material_gained
        
        if VERBOSE:
            move_str = f"{move_num}." if is_white else f"{move_num}..."
            print(f"\nChecking move {move} for hanging pieces:")
            print(f"Our color: {'White' if board.turn else 'Black'}")
            print(f"Is capture: {captured_piece is not None}")
            if captured_piece:
                print(f"Captured piece: {captured_piece.symbol()}")
            
            # Print details about hanging pieces
            for square in chess.SQUARES:
                piece = board_copy.piece_at(square)
                if piece and piece.color == board.turn:
                    result, piece_obj, attackers = self.is_piece_undefended_and_attacked(board_copy, square, board.turn)
                    if result:
                        square_name = chess.square_name(square)
                        print(f"Debug - Square: {square_name}, Piece: {'White' if piece.color else 'Black'} {piece.symbol()}")
                        print(f"Debug - Completely undefended piece under attack")
                        print(f"Debug - Attackers ({len(attackers)}): {[chess.square_name(sq) for sq in attackers]}")
                        print(f"Debug - Actually vulnerable: {result}")
        
        if net_material_loss <= 0:  # If we don't lose material, it's a safe move
            if VERBOSE:
                move_str = f"{move_num}." if is_white else f"{move_num}..."
                print(f"Move {move_str} Selected candidate #{candidate_num}: {move}")
            return True
        else:
            if VERBOSE:
                move_str = f"{move_num}." if is_white else f"{move_num}..."
                print(f"Move {move_str} Candidate #{candidate_num}: Skipping move {move} as it leaves {most_valuable_hanging} points of material hanging (most valuable piece)")
                print(f"  Total material at risk: {total_hanging_value} points, but opponent can only take {most_valuable_hanging} points immediately:")
                # Print details of each hanging piece
                for square in chess.SQUARES:
                    piece = board_copy.piece_at(square)
                    if piece and piece.color == board.turn:
                        result, piece_obj, attackers = self.is_piece_undefended_and_attacked(board_copy, square, board.turn)
                        if result:
                            square_name = chess.square_name(square)
                            piece_value = self.get_piece_value(piece_obj)
                            marker = " ← MOST VALUABLE" if piece_value == most_valuable_hanging else ""
                            print(f"  - {piece.symbol()} on {square_name} ({len(attackers)} attackers vs 0 defenders): {[chess.square_name(sq) for sq in attackers]}{marker}")
            # Store the move and its material loss for later comparison
            rejected_moves.append((move, net_material_loss, total_hanging_value, candidate_num))
            return False

def play_game(white_player, black_player, num_games: int = 1) -> Tuple[int, int, int]:
    wins_white = 0
    wins_black = 0
    draws = 0
    
    # Additional statistics
    early_wins_white = 0  # Wins in first 20 moves
    early_wins_black = 0
    mid_wins_white = 0   # Wins in moves 20-50
    mid_wins_black = 0
    late_wins_white = 0  # Wins after move 50
    late_wins_black = 0
    
    # Position evaluation tracking (from white's perspective)
    total_position_eval = 0
    total_positions = 0
    
    # Centipawn loss tracking
    total_cpl_model = 0
    total_cpl_stockfish = 0
    total_positions_model = 0
    total_positions_stockfish = 0
    
    # Create a Stockfish instance for evaluation at full strength
    with chess.engine.SimpleEngine.popen_uci('/usr/games/stockfish') as analysis_engine:
        analysis_engine.configure({"Threads": 1})  # Use single thread for analysis
        
        for game_num in range(num_games):
            board = chess.Board()
            move_count = 0
            game_moves = []  # Store moves for PGN
            
            # Track evaluations for this game
            game_total_eval = 0
            game_positions = 0
            total_cpl_this_game_model = 0
            total_cpl_this_game_stockfish = 0
            positions_evaluated_model = 0
            positions_evaluated_stockfish = 0
            
            print(f"\nPlaying game {game_num + 1}/{num_games}")
            
            while not board.is_game_over():
                current_move_num = (move_count // 2) + 1
                
                # Get position evaluation before move from full strength Stockfish
                try:
                    info = analysis_engine.analyse(board, chess.engine.Limit(time=0.001))
                    current_eval = info["score"].relative.score(mate_score=1000) / 100.0  # Convert to pawns
                    # Clip evaluation between -10 and 10
                    current_eval = max(-20, min(20, current_eval))
                    game_total_eval += current_eval
                    game_positions += 1
                except:
                    current_eval = 0
                
                if board.turn:  # White's turn (Model's turn)
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
                            
                            # Calculate centipawn loss for model
                            try:
                                # Get best move evaluation from full strength Stockfish
                                best_move_info = analysis_engine.analyse(board, chess.engine.Limit(time=0.001))
                                best_eval = best_move_info["score"].relative.score(mate_score=1000) / 100.0
                                
                                # Calculate centipawn loss
                                cpl = max(0, (best_eval - (-current_eval)) * 100)  # Convert to centipawns
                                total_cpl_model += cpl
                                total_cpl_this_game_model += cpl
                                positions_evaluated_model += 1
                                total_positions_model += 1
                            except:
                                pass
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
                        
                        # Calculate centipawn loss for Stockfish
                        try:
                            # Get best move evaluation from full strength Stockfish
                            best_move_info = analysis_engine.analyse(board, chess.engine.Limit(time=0.001))
                            best_eval = best_move_info["score"].relative.score(mate_score=1000) / 100.0
                            
                            # Calculate centipawn loss
                            cpl = max(0, (best_eval - current_eval) * 100)  # Convert to centipawns
                            total_cpl_stockfish += cpl
                            total_cpl_this_game_stockfish += cpl
                            positions_evaluated_stockfish += 1
                            total_positions_stockfish += 1
                        except:
                            pass
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
            
            # Update total position evaluation
            total_position_eval += game_total_eval
            total_positions += game_positions
            
            result = board.outcome()
            if result:
                if result.winner == chess.WHITE:
                    wins_white += 1
                    # Categorize win by game length
                    if move_count <= 40:  # 20 moves (each player moves once per move number)
                        early_wins_white += 1
                    elif move_count <= 100:  # 50 moves
                        mid_wins_white += 1
                    else:
                        late_wins_white += 1
                    print(f"Game {game_num + 1}: White wins")
                elif result.winner == chess.BLACK:
                    wins_black += 1
                    # Categorize win by game length
                    if move_count <= 40:
                        early_wins_black += 1
                    elif move_count <= 100:
                        mid_wins_black += 1
                    else:
                        late_wins_black += 1
                    print(f"Game {game_num + 1}: Black wins")
                else:
                    draws += 1
                    print(f"Game {game_num + 1}: Draw")
            else:
                draws += 1
                print(f"Game {game_num + 1}: Draw (no valid moves)")
            
            # Print per-game statistics
            if VERBOSE:
                print(f"Average position evaluation this game: {game_total_eval/game_positions if game_positions > 0 else 0:.2f}")
                print(f"Average centipawn loss this game - Model: {total_cpl_this_game_model/positions_evaluated_model if positions_evaluated_model > 0 else 0:.2f}")
                print(f"Average centipawn loss this game - Stockfish: {total_cpl_this_game_stockfish/positions_evaluated_stockfish if positions_evaluated_stockfish > 0 else 0:.2f}")
    
    # Calculate averages
    avg_position_eval = total_position_eval / total_positions if total_positions > 0 else 0
    avg_cpl_model = total_cpl_model / total_positions_model if total_positions_model > 0 else 0
    avg_cpl_stockfish = total_cpl_stockfish / total_positions_stockfish if total_positions_stockfish > 0 else 0
    
    # Return extended statistics
    return (wins_white, wins_black, draws,
            early_wins_white, early_wins_black,
            mid_wins_white, mid_wins_black,
            late_wins_white, late_wins_black,
            avg_position_eval,
            avg_cpl_model, avg_cpl_stockfish)

def main():
    # Configuration variables - uncomment 1 of the 3 modes to be the opponent
    # mode = 'model' #used if you want to play against another trained model
    mode = 'stockfish' #used if you want to play against stockfish
    # mode = 'random' #used if you want to play against random moves
    
    # Model configuration
    # model_type = 'categorical'  # 'categorical', 'spatial', or 'attention'
    # model_path = 'models/trainv3_model_architecture_v2_epochs250.pth'  # Path to your main model
    # mapping_path = "models/move_to_int_architecture_v2"  # Only needed for categorical models
    
    # For spatial/attention models, use these instead:
    model_type = 'attention'  # Choose between 'spatial' or 'attention'
    model_path = 'models/attention_model_checkpoint_epoch_10.pth'  # Path to your model
    mapping_path = None  # Not needed for spatial/attention models
    
    
    # Opponent model configuration
    opponent_model_path = 'models/trainv3_model_architecture_v2_epochs250.pth'  # Path to opponent model (for mode='model')
    opponent_model_type = 'categorical'  # Type of opponent model ('categorical', 'spatial', or 'attention')
    opponent_mapping_path = "models/move_to_int_architecture_v2"  # Mapping for opponent model (if categorical)
    
    # Stockfish configuration
    stockfish_path = '/usr/games/stockfish'  # Path to stockfish executable (for mode='stockfish')
    stockfish_elo = 1350  # Stockfish ELO rating (only used for opponent, not for statistics) - minimum 1350
    time_limit = 0.01  # Time limit per move for Stockfish (in seconds)
    
    # Game configuration
    num_games = 100  # Number of games to play
    
    # Initialize main model
    if model_type == 'categorical':
        main_model = ChessEvaluator(model_path, mapping_path, model_type='categorical')
    elif model_type in ['spatial', 'attention']:
        main_model = ChessEvaluator(model_path, model_type=model_type)
    else:
        raise ValueError("model_type must be either 'categorical', 'spatial', or 'attention'")
    
    print(f"Loaded {model_type} model from {model_path}")
    
    if mode == 'random':
        # Play against random moves
        if VERBOSE:
            print("\nEvaluating against random moves...")
        stats = play_game(main_model, "random", num_games)
        opponent_desc = "Random"
        
    elif mode == 'model':
        if not opponent_model_path:
            raise ValueError("opponent_model_path must be set for model mode")
        
        # Play against another model
        if VERBOSE:
            print("\nEvaluating against another model...")
        
        if opponent_model_type == 'categorical':
            opponent_model = ChessEvaluator(opponent_model_path, opponent_mapping_path, model_type='categorical')
        elif opponent_model_type in ['spatial', 'attention']:
            opponent_model = ChessEvaluator(opponent_model_path, model_type=opponent_model_type)
        else:
            raise ValueError("opponent_model_type must be either 'categorical', 'spatial', or 'attention'")
            
        stats = play_game(main_model, opponent_model, num_games)
        opponent_desc = f"Opponent Model ({opponent_model_type})"
        
    elif mode == 'stockfish':
        if not stockfish_path:
            raise ValueError("stockfish_path must be set for stockfish mode")
        
        # Play against Stockfish
        if VERBOSE:
            print(f"\nEvaluating against Stockfish (ELO: {stockfish_elo})...")
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})  # Set ELO rating
        
        stats = play_game(main_model, engine, num_games)
        engine.quit()
        opponent_desc = f"Stockfish (ELO {stockfish_elo})"
    
    else:
        raise ValueError("Invalid mode. Choose from: 'random', 'model', or 'stockfish'")
    
    # Unpack statistics
    (wins_white, wins_black, draws,
     early_wins_white, early_wins_black,
     mid_wins_white, mid_wins_black,
     late_wins_white, late_wins_black,
     avg_position_eval,
     avg_cpl_model, avg_cpl_stockfish) = stats
    
    # Print results (these always show regardless of VERBOSE setting)
    print("\nEvaluation Results:")
    print(f"Main model type: {model_type}")
    print(f"Total games: {num_games}")
    print("\nOverall Results:")
    print(f"Model wins: {(wins_white / num_games) * 100:.1f}%")
    print(f"{opponent_desc} wins: {(wins_black / num_games) * 100:.1f}%")
    print(f"Draws: {(draws / num_games) * 100:.1f}%")
    
    # Print game length statistics
    print("\nGame Length Analysis:")
    total_wins = wins_white + wins_black
    if total_wins > 0:  # Avoid division by zero
        print("Early game results (first 20 moves):")
        print(f"  Model wins: {(early_wins_white / total_wins) * 100:.1f}%")
        print(f"  {opponent_desc} wins: {(early_wins_black / total_wins) * 100:.1f}%")
        
        print("\nMid game results (moves 20-50):")
        print(f"  Model wins: {(mid_wins_white / total_wins) * 100:.1f}%")
        print(f"  {opponent_desc} wins: {(mid_wins_black / total_wins) * 100:.1f}%")
        
        print("\nLate game results (after move 50):")
        print(f"  Model wins: {(late_wins_white / total_wins) * 100:.1f}%")
        print(f"  {opponent_desc} wins: {(late_wins_black / total_wins) * 100:.1f}%")
    
    # Print evaluation statistics
    print("\nAverage Position Evaluation (from White's perspective):")
    print(f"  {avg_position_eval:.2f} pawns")
    
    # Print centipawn loss statistics
    print("\nAverage Centipawn Loss:")
    print(f"  Model: {avg_cpl_model:.1f}")
    print(f"  {opponent_desc}: {avg_cpl_stockfish:.1f}")

if __name__ == "__main__":
    main() 