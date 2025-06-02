import numpy as np
from chess import Board
import chess

def board_to_matrix(board: Board):
    # 8x8 is a size of the chess board.
    # 12 = number of unique pieces.
    # 13th board for legal moves (WHERE we can move)
    # maybe 14th for squares FROM WHICH we can move? idk
    matrix = np.zeros((13, 8, 8))
    piece_map = board.piece_map()

    # Populate first 12 8x8 boards (where pieces are)
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1

    # Populate the legal moves board (13th 8x8 board)
    legal_moves = board.legal_moves
    for move in legal_moves:
        to_square = move.to_square
        row_to, col_to = divmod(to_square, 8)
        matrix[12, row_to, col_to] = 1

    return matrix


def create_input_for_nn(games):
    X = []
    y = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_matrix(board))
            y.append(move.uci())
            board.push(move)
    return np.array(X, dtype=np.float32), np.array(y)


def encode_moves_spatial(moves):
    """
    Encode moves as spatial 2-channel 8x8 representations.
    Channel 0: from square (1 at source position, 0 elsewhere)
    Channel 1: to square (1 at destination position, 0 elsewhere)
    
    Args:
        moves: List of UCI move strings (e.g., ['e2e4', 'e7e5', ...])
    
    Returns:
        numpy array of shape (num_moves, 2, 8, 8)
    """
    encoded_moves = np.zeros((len(moves), 2, 8, 8), dtype=np.float32)
    
    for i, move_uci in enumerate(moves):
        try:
            move = chess.Move.from_uci(move_uci)
            
            # From square (channel 0)
            from_row, from_col = divmod(move.from_square, 8)
            encoded_moves[i, 0, from_row, from_col] = 1.0
            
            # To square (channel 1)
            to_row, to_col = divmod(move.to_square, 8)
            encoded_moves[i, 1, to_row, to_col] = 1.0
            
        except ValueError:
            # If invalid move, leave as zeros
            continue
    
    return encoded_moves


def decode_move_spatial(prediction, legal_moves):
    """
    Decode spatial move prediction back to UCI move string.
    
    Args:
        prediction: numpy array of shape (2, 8, 8) - softmax probabilities
        legal_moves: list of legal chess.Move objects
        
    Returns:
        string: UCI move or None if no valid move found
    """
    # Get the most likely from and to squares
    from_probs = prediction[0].flatten()
    to_probs = prediction[1].flatten()
    
    # Convert legal moves to (from_square, to_square) tuples for quick lookup
    legal_move_tuples = [(move.from_square, move.to_square) for move in legal_moves]
    legal_moves_dict = {(move.from_square, move.to_square): move.uci() for move in legal_moves}
    
    # Try combinations in order of probability
    from_indices = np.argsort(from_probs)[::-1]  # Sort descending
    to_indices = np.argsort(to_probs)[::-1]      # Sort descending
    
    # Try the most probable combinations first
    for from_idx in from_indices[:10]:  # Top 10 from squares
        for to_idx in to_indices[:10]:  # Top 10 to squares
            if (from_idx, to_idx) in legal_move_tuples:
                return legal_moves_dict[(from_idx, to_idx)]
    
    # If no valid combination found in top candidates, return first legal move
    if legal_moves:
        return legal_moves[0].uci()
    
    return None


# Keep the old categoricalfunction for backward compatibility
def encode_moves(moves):
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return np.array([move_to_int[move] for move in moves], dtype=np.float32), move_to_int