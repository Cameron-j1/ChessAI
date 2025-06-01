from chess import Board, pgn
from other_functions import board_to_matrix
import torch
from chess_model_class import ChessModel
import pickle
import numpy as np
import chess

def prepare_input(board: Board):
    matrix = board_to_matrix(board)
    X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    return X_tensor

def clear_terminal():
    import os
    os.system('clear' if os.name == 'posix' else 'cls')

def get_player_move(board: Board) -> str:
    while True:
        try:
            move = input("\nEnter your move (in UCI format, e.g., 'e2e4'): ").strip()
            if move.lower() == 'quit':
                return 'quit'
            if move.lower() == 'help':
                print("\nMove format examples:")
                print("- Regular moves: e2e4, d7d5")
                print("- Castling: e1g1 (king-side), e1c1 (queen-side)")
                print("- Pawn promotion: e7e8q (promote to queen)")
                print("Type 'quit' to end the game")
                continue
            
            # Validate the move
            try:
                if move not in [m.uci() for m in board.legal_moves]:
                    print("\nIllegal move! Please try again.")
                    print("Type 'help' for move format examples.")
                    continue
                return move
            except ValueError:
                print("\nInvalid move format! Please try again.")
                print("Type 'help' for move format examples.")
        except KeyboardInterrupt:
            return 'quit'

def predict_move(board: Board):
    X_tensor = prepare_input(board).to(device)
    
    with torch.no_grad():
        logits = model(X_tensor)
    
    logits = logits.squeeze(0)  # Remove batch dimension
    
    probabilities = torch.softmax(logits, dim=0).cpu().numpy()
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    sorted_indices = np.argsort(probabilities)[::-1]
    
    for move_index in sorted_indices:
        move = int_to_move[move_index]
        if move in legal_moves_uci:
            return move
    
    # If no valid move found, return a random legal move
    if legal_moves:
        return np.random.choice(legal_moves).uci()
    return None

def print_game_status(board: Board, player_color: str):
    clear_terminal()
    print("\nCurrent board position:")
    if player_color == "black":
        print(board.transform(chess.flip_vertical))
    else:
        print(board)
    
    if board.is_checkmate():
        print("\nCheckmate! Game Over!")
        return True
    elif board.is_stalemate():
        print("\nStalemate! Game Over!")
        return True
    elif board.is_insufficient_material():
        print("\nDraw due to insufficient material! Game Over!")
        return True
    elif board.is_check():
        print("\nCheck!")
    return False

def main():
    # Load the model and mappings
    print("Loading AI model...")
    
    try:
        with open("models/move_to_int_architecture_v2", "rb") as file:
            global move_to_int
            move_to_int = pickle.load(file)
    except FileNotFoundError:
        print("Error: Could not find the move mapping file!")
        return

    global int_to_move
    int_to_move = {v: k for k, v in move_to_int.items()}

    # Initialize model
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    global model
    model = ChessModel(num_classes=len(move_to_int))
    try:
        model.load_state_dict(torch.load("models/model_architecture_v2_epochs20.pth"))
    except FileNotFoundError:
        print("Error: Could not find the model file!")
        return
        
    model.to(device)
    model.eval()

    # Game setup
    while True:
        color_choice = input("\nDo you want to play as white or black? (w/b): ").lower()
        if color_choice in ['w', 'b']:
            break
        print("Please enter 'w' for white or 'b' for black.")

    player_color = "white" if color_choice == 'w' else "black"
    board = Board()
    
    print("\nGame starting!")
    print("Type 'help' for move format examples or 'quit' to end the game.")

    # Main game loop
    while not board.is_game_over():
        if print_game_status(board, player_color):
            break

        current_turn = "white" if board.turn else "black"
        
        if current_turn == player_color:
            # Player's turn
            move = get_player_move(board)
            if move == 'quit':
                print("\nGame ended by player.")
                break
        else:
            # AI's turn
            print("\nAI is thinking...")
            move = predict_move(board)
            if not move:
                print("\nAI couldn't find a valid move! Game Over!")
                break
            print(f"AI plays: {move}")

        board.push_uci(move)

    # Game ended, show final position and game PGN
    print("\nFinal position:")
    print(board)
    print("\nGame PGN:")
    print(str(pgn.Game.from_board(board)))

if __name__ == "__main__":
    main()

