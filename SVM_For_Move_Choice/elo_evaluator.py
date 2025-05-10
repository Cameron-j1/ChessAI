import chess
import chess.engine
import random
import csv
import os
from chess_svm_model import ChessMoveEvaluator

# Configuration
STOCKFISH_PATH = "/usr/games/stockfish"  # Change as needed
MODEL_PATH = "chess_svm_model.pkl"
TIME_LIMITS = []# [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]  # Simulate lower Elo
GAMES_PER_CONFIG = 300
RESULT_CSV = "svm_vs_stockfish_scaled_results.csv"


def play_vs_stockfish_time_limit(evaluator, time_limit):
    board = chess.Board()
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        engine.configure({"UCI_LimitStrength": False})  # Full strength, limited by time

        while not board.is_game_over():
            if board.turn == chess.WHITE:
                move, _ = evaluator.find_best_move(board)
            else:
                result = engine.play(board, chess.engine.Limit(time=time_limit))
                move = result.move

            if move is None or move not in board.legal_moves:
                break
            board.push(move)

    return board.outcome()


def play_vs_random(evaluator):
    board = chess.Board()

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move, _ = evaluator.find_best_move(board)
        else:
            move = random.choice(list(board.legal_moves))

        board.push(move)

    return board.outcome()


def main():
    evaluator = ChessMoveEvaluator(stockfish_path=STOCKFISH_PATH)
    evaluator.load_model(MODEL_PATH)

    results = []

    # Varying Stockfish time limits (simulate weak opponents)
    for time_limit in TIME_LIMITS:
        wins, draws, losses = 0, 0, 0
        print(f"Testing vs Stockfish (time={time_limit:.2f}s)...")
        for _ in range(GAMES_PER_CONFIG):
            outcome = play_vs_stockfish_time_limit(evaluator, time_limit)
            if outcome.winner is None:
                draws += 1
            elif outcome.winner == chess.WHITE:
                wins += 1
            else:
                losses += 1
        results.append({
            "opponent": f"Stockfish_t{time_limit:.2f}s",
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": wins / GAMES_PER_CONFIG,
            "draw_rate": draws / GAMES_PER_CONFIG,
            "loss_rate": losses / GAMES_PER_CONFIG
        })

    # Random move opponent
    print("Testing vs Random Move Opponent...")
    wins, draws, losses = 0, 0, 0
    for i in range(GAMES_PER_CONFIG):
        print(i)
        outcome = play_vs_random(evaluator)
        if outcome.winner is None:
            draws += 1
        elif outcome.winner == chess.WHITE:
            wins += 1
        else:
            losses += 1
    results.append({
        "opponent": "Random",
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / GAMES_PER_CONFIG,
        "draw_rate": draws / GAMES_PER_CONFIG,
        "loss_rate": losses / GAMES_PER_CONFIG
    })

    # Save to CSV
    with open(RESULT_CSV, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {RESULT_CSV}")


if __name__ == "__main__":
    main()
