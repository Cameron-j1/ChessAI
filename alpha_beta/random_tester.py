import chess
import time
import random
from svm_position_evaluator import ChessPositionEvaluator
from chess_engine import ChessEngine  # Ensure this matches your engine filename

def play_vs_random(engine, max_moves=100, play_as_white=True, verbose=False):
    """
    Play a game where the engine plays against a random move player.

    Args:
        engine: ChessEngine instance
        max_moves: Max half-moves before draw
        play_as_white: If True, engine plays white, else black
        verbose: If True, print board and moves

    Returns:
        str: Result: "win", "loss", "draw"
    """
    board = chess.Board()
    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        if verbose:
            print(f"\nMove {move_count + 1} ({'White' if board.turn else 'Black'})")
            print(board)

        if (board.turn == chess.WHITE and play_as_white) or (board.turn == chess.BLACK and not play_as_white):
            move = engine.get_best_move(board, time_limit=3.0)
            if verbose:
                print(f"Engine played: {board.san(move)}")
        else:
            move = random.choice(list(board.legal_moves))
            if verbose:
                print(f"Random played: {board.san(move)}")

        board.push(move)
        move_count += 1

    if verbose:
        print("\nFinal position:")
        print(board)

    if board.is_checkmate():
        winner = "white" if board.turn == chess.BLACK else "black"
        return "win" if (winner == "white") == play_as_white else "loss"
    elif board.is_stalemate() or board.is_insufficient_material() or \
         board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return "draw"
    elif move_count >= max_moves:
        return "draw"
    return "draw"

def run_benchmark(games=10, play_as_white=True):
    model_path = "chess_position_evaluator_5000.pkl"
    stockfish_path = "/usr/games/stockfish"

    evaluator = ChessPositionEvaluator(stockfish_path=stockfish_path)
    try:
        evaluator.load_model(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Model load failed: {e}")
        evaluator = None

    engine = ChessEngine(position_evaluator=evaluator, stockfish_path=stockfish_path, search_depth=4)

    results = {"win": 0, "loss": 0, "draw": 0}

    for i in range(games):
        print(f"\n=== Game {i + 1} ===")
        result = play_vs_random(engine, max_moves=100, play_as_white=play_as_white, verbose=False)
        print(f"Result: {result}")
        results[result] += 1

    print("\n=== Summary ===")
    print(f"Games played: {games}")
    print(f"Wins:   {results['win']}")
    print(f"Losses: {results['loss']}")
    print(f"Draws:  {results['draw']}")
    win_rate = results['win'] / games * 100
    print(f"Win Rate: {win_rate:.1f}%")

if __name__ == "__main__":
    run_benchmark(games=20, play_as_white=True)
