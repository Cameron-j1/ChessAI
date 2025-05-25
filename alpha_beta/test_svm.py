import chess.pgn
import numpy as np
from svm_position_evaluator import ChessPositionEvaluator


def test_model_on_pgn(evaluator, pgn_file, num_games=10, time_limit=0.01):
    """
    Evaluate the trained model on a new PGN file and print average prediction error.

    Args:
        evaluator (ChessPositionEvaluator): Initialized and trained evaluator
        pgn_file (str): Path to PGN file
        num_games (int): Number of games to evaluate
        time_limit (float): Stockfish time per position (in seconds)

    Returns:
        float: Mean absolute error across tested positions
    """
    if evaluator.model is None:
        raise ValueError("Model has not been trained or loaded")

    actual_scores = []
    predicted_scores = []

    with chess.engine.SimpleEngine.popen_uci(evaluator.stockfish_path) as engine:
        engine.configure({"Threads": 4, "Hash": 128})
        with open(pgn_file) as pgn:
            games_evaluated = 0
            while games_evaluated < num_games:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break

                board = game.board()
                move_count = 0
                for move in game.mainline_moves():
                    move_count += 1
                    if move_count % 3 != 0:
                        board.push(move)
                        continue

                    # Get true evaluation
                    true_eval = evaluator._get_stockfish_evaluation(board, engine, time_limit=time_limit)

                    # Get model prediction
                    try:
                        pred_eval = evaluator.evaluate_position(board)
                    except Exception as e:
                        print(f"Prediction failed: {e}")
                        continue

                    actual_scores.append(true_eval)
                    predicted_scores.append(pred_eval)

                    board.push(move)

                games_evaluated += 1

    if not actual_scores:
        print("No positions were evaluated.")
        return None

    actual_scores = np.array(actual_scores)
    predicted_scores = np.array(predicted_scores)
    mae = np.mean(np.abs(actual_scores - predicted_scores))

    print(f"Tested {len(actual_scores)} positions from {games_evaluated} games.")
    print(f"Mean Absolute Error: {mae:.4f} pawns")

    return mae


if __name__ == "__main__":
    # You can change this if needed
    model_path = "chess_position_evaluator_5000.pkl"
    pgn_path = "standardover2000-2021.pgn"

    evaluator = ChessPositionEvaluator(stockfish_path="/usr/games/stockfish", use_gpu=True)
    evaluator.load_model(model_path)

    test_model_on_pgn(evaluator, pgn_path, num_games=20)