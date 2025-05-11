import chess
import chess.engine
import random
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from chess_svm_model import ChessMoveEvaluator

# Configuration
STOCKFISH_PATH = "/usr/games/stockfish"
MODEL_PATH = "test.pkl"
TIME_LIMITS = []  # e.g., [0.01, 0.02, 0.05, 0.1, 0.2]
GAMES_PER_CONFIG = 300
NUM_THREADS = 50
RESULT_CSV = "chess_svm_model_700_games_featuresV3_gpu_vs_random.csv"

def _init_evaluator():
    """ Each thread gets its own evaluator instance """
    evaluator = ChessMoveEvaluator(stockfish_path=STOCKFISH_PATH)
    evaluator.load_model(MODEL_PATH)
    return evaluator

def play_vs_stockfish_time_limit(_, time_limit):
    evaluator = _init_evaluator()
    board = chess.Board()
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        engine.configure({"UCI_LimitStrength": False})

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

def play_vs_random(_):
    evaluator = _init_evaluator()
    board = chess.Board()
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move, _ = evaluator.find_best_move(board)
        else:
            move = random.choice(list(board.legal_moves))
        board.push(move)
    return board.outcome()

def run_games_parallel(play_fn, games, **kwargs):
    wins = draws = losses = 0
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(play_fn, i, **kwargs) for i in range(games)]
        for i, future in enumerate(tqdm(as_completed(futures), total=games, desc="Games")):
            outcome = future.result()
            if outcome.winner is None:
                draws += 1
            elif outcome.winner == chess.WHITE:
                wins += 1
            else:
                losses += 1
    return wins, draws, losses

def main():
    results = []

    # Time-limited Stockfish opponents
    for time_limit in TIME_LIMITS:
        print(f"\n==> Running {GAMES_PER_CONFIG} games vs Stockfish (time_limit={time_limit:.2f}s) using {NUM_THREADS} threads...")
        wins, draws, losses = run_games_parallel(play_vs_stockfish_time_limit, GAMES_PER_CONFIG, time_limit=time_limit)
        results.append({
            "opponent": f"Stockfish_t{time_limit:.2f}s",
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": wins / GAMES_PER_CONFIG,
            "draw_rate": draws / GAMES_PER_CONFIG,
            "loss_rate": losses / GAMES_PER_CONFIG
        })

    # Random opponent
    print(f"\n==> Running {GAMES_PER_CONFIG} games vs Random Move Opponent using {NUM_THREADS} threads...")
    wins, draws, losses = run_games_parallel(play_vs_random, GAMES_PER_CONFIG)
    results.append({
        "opponent": "Random",
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / GAMES_PER_CONFIG,
        "draw_rate": draws / GAMES_PER_CONFIG,
        "loss_rate": losses / GAMES_PER_CONFIG
    })

    # Save results
    with open(RESULT_CSV, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {RESULT_CSV}")

if __name__ == "__main__":
    main()
