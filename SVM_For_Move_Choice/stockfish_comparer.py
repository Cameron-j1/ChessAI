#!/usr/bin/env python3
"""
Stockfish Evaluation Time Analysis

This script analyzes how Stockfish's evaluations vary with different evaluation times.
It loads positions from PGN files, evaluates each position with different times,
and compares the results against a high-quality baseline evaluation.
"""

import chess
import chess.engine
import chess.pgn
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd


class StockfishEvaluator:
    def __init__(self, stockfish_path: str = "stockfish"):
        """Initialize Stockfish engine."""
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    def evaluate(self, board: chess.Board, eval_time: float) -> float:
        """
        Evaluate a position using Stockfish with the given time limit.
        Returns evaluation in pawns (positive = white advantage).
        """
        info = self.engine.analyse(board, chess.engine.Limit(time=eval_time))
        cp_score = info["score"].white().score(mate_score=10000)
        return cp_score / 100.0  # Convert centipawns to pawns
    
    def get_best_move(self, board: chess.Board, eval_time: float) -> chess.Move:
        """Get the best move according to Stockfish."""
        result = self.engine.play(board, chess.engine.Limit(time=eval_time))
        return result.move
    
    def close(self):
        """Properly close the engine."""
        self.engine.quit()


def load_positions_from_pgn(pgn_file: str, max_positions: int = 100) -> List[chess.Board]:
    """
    Load positions from a PGN file.
    Returns a list of unique board positions.
    """
    positions = []
    unique_fens = set()
    
    try:
        with open(pgn_file) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:  # End of file
                    break
                
                # Process this game
                board = game.board()
                for move in game.mainline_moves():
                    board.push(move)
                    
                    # To avoid duplicate positions, use FEN
                    fen = board.fen().split(' ')[0]  # Only position part
                    if fen not in unique_fens:
                        unique_fens.add(fen)
                        positions.append(board.copy())
                    
                    if len(positions) >= max_positions:
                        return positions
    except Exception as e:
        print(f"Error loading PGN file: {e}")
    
    return positions


def analyze_positions(
    positions: List[chess.Board],
    eval_times: List[float],
    baseline_time: float,
    stockfish_path: str = "stockfish"
) -> Dict:
    evaluator = StockfishEvaluator(stockfish_path)
    results = {
        "baseline_evals": [],
        "time_evals": {t: [] for t in eval_times},
        "eval_times": eval_times,
        "baseline_time": baseline_time,
    }

    total_positions = len(positions)

    for i, board in enumerate(positions):
        print(f"Evaluating position {i+1}/{total_positions}", end='\r')

        baseline_eval = evaluator.evaluate(board, baseline_time)
        results["baseline_evals"].append(baseline_eval)

        for eval_time in eval_times:
            eval_result = evaluator.evaluate(board, eval_time)
            results["time_evals"][eval_time].append(eval_result)

    evaluator.close()
    print(f"\nEvaluated {total_positions} positions")
    return results


def calculate_metrics(results: Dict) -> Dict:
    """Calculate error metrics between baseline and different eval times."""
    metrics = {
        "mae": {},  # Mean Absolute Error
        "rmse": {},  # Root Mean Squared Error
        "max_error": {},  # Maximum Error
        # "move_accuracy": {},  # Best move accuracy
        "eval_time_ratio": {},  # Ratio of eval time to baseline
    }
    
    baseline_evals = np.array(results["baseline_evals"])
    
    for eval_time in results["eval_times"]:
        evals = np.array(results["time_evals"][eval_time])
        errors = evals - baseline_evals
        abs_errors = np.abs(errors)
        
        metrics["mae"][eval_time] = np.mean(abs_errors)
        metrics["rmse"][eval_time] = np.sqrt(np.mean(np.square(errors)))
        metrics["max_error"][eval_time] = np.max(abs_errors)
        # metrics["move_accuracy"][eval_time] = np.mean(results["best_move_accuracy"][eval_time]) * 100
        metrics["eval_time_ratio"][eval_time] = eval_time / results["baseline_time"]
    
    return metrics


def plot_results(metrics: Dict, output_file: Optional[str] = None):
    """Generate plots from the analysis results."""
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Stockfish Evaluation Time Analysis\nBaseline: {metrics['eval_time_ratio'][min(metrics['eval_time_ratio'].keys())]:.1f}s", fontsize=16)
    
    eval_times = sorted(metrics["mae"].keys())
    time_ratios = [metrics["eval_time_ratio"][t] for t in eval_times]
    
    # Plot MAE (Mean Absolute Error)
    ax1 = axes[0, 0]
    ax1.plot(eval_times, [metrics["mae"][t] for t in eval_times], 'o-', color='blue')
    ax1.set_xscale('log')
    ax1.set_xlabel('Evaluation Time (s)')
    ax1.set_ylabel('Mean Absolute Error (pawns)')
    ax1.set_title('Mean Absolute Error vs Evaluation Time')
    ax1.grid(True)
    
    # Plot Move Accuracy
    ax2 = axes[0, 1]
    ax2.plot(eval_times, [metrics["move_accuracy"][t] for t in eval_times], 'o-', color='green')
    ax2.set_xscale('log')
    ax2.set_xlabel('Evaluation Time (s)')
    ax2.set_ylabel('Move Accuracy (%)')
    ax2.set_title('Best Move Accuracy vs Evaluation Time')
    ax2.grid(True)
    
    # Plot Max Error
    ax3 = axes[1, 0]
    ax3.plot(eval_times, [metrics["max_error"][t] for t in eval_times], 'o-', color='red')
    ax3.set_xscale('log')
    ax3.set_xlabel('Evaluation Time (s)')
    ax3.set_ylabel('Maximum Error (pawns)')
    ax3.set_title('Maximum Error vs Evaluation Time')
    ax3.grid(True)
    
    # Plot RMSE
    ax4 = axes[1, 1]
    ax4.plot(eval_times, [metrics["rmse"][t] for t in eval_times], 'o-', color='purple')
    ax4.set_xscale('log')
    ax4.set_xlabel('Evaluation Time (s)')
    ax4.set_ylabel('RMSE (pawns)')
    ax4.set_title('Root Mean Squared Error vs Evaluation Time')
    ax4.grid(True)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def save_results_to_csv(results: Dict, metrics: Dict, output_file: str):
    """Save detailed results to a CSV file."""
    # Create a DataFrame with all results
    data = {
        "baseline_time": results["baseline_time"],
        "baseline_eval": results["baseline_evals"],
    }
    
    # Add evaluation results for each time
    for eval_time in results["eval_times"]:
        data[f"eval_{eval_time}s"] = results["time_evals"][eval_time]
        data[f"error_{eval_time}s"] = np.array(results["time_evals"][eval_time]) - np.array(results["baseline_evals"])
        data[f"best_move_match_{eval_time}s"] = results["best_move_accuracy"][eval_time]
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Detailed results saved to {output_file}")
    
    # Also save the metrics summary
    metrics_data = {
        "eval_time": [],
        "eval_time_ratio": [],
        "mae": [],
        "rmse": [],
        "max_error": [],
        "move_accuracy": [],
    }
    
    for eval_time in sorted(metrics["mae"].keys()):
        metrics_data["eval_time"].append(eval_time)
        metrics_data["eval_time_ratio"].append(metrics["eval_time_ratio"][eval_time])
        metrics_data["mae"].append(metrics["mae"][eval_time])
        metrics_data["rmse"].append(metrics["rmse"][eval_time])
        metrics_data["max_error"].append(metrics["max_error"][eval_time])
        metrics_data["move_accuracy"].append(metrics["move_accuracy"][eval_time])
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_output = output_file.replace(".csv", "_metrics.csv")
    metrics_df.to_csv(metrics_output, index=False)
    print(f"Metrics summary saved to {metrics_output}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Stockfish evaluation time performance")
    parser.add_argument("--pgn", type=str, help="Path to PGN file", default="standard16klinesover2000.pgn")
    parser.add_argument("--stockfish", type=str, default="/usr/games/stockfish", help="Path to Stockfish engine")
    parser.add_argument("--baseline-time", type=float, default=1.0, 
                      help="Baseline evaluation time in seconds (default: 1.0)")
    parser.add_argument("--eval-times", type=str, default="0.001,0.005,0.01,0.05,0.1,0.2,0.5", 
                      help="Comma-separated list of evaluation times to test")
    parser.add_argument("--max-positions", type=int, default=1000, 
                      help="Maximum number of positions to analyze (default: 100)")
    parser.add_argument("--output-prefix", type=str, default="stockfish_analysis", 
                      help="Prefix for output files")
    
    args = parser.parse_args()
    
    # Parse evaluation times
    eval_times = [float(t) for t in args.eval_times.split(",")]
    
    # Create output directory if it doesn't exist
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Load positions from PGN
    print(f"Loading positions from {args.pgn}...")
    positions = load_positions_from_pgn(args.pgn, args.max_positions)
    print(f"Loaded {len(positions)} unique positions")
    
    if not positions:
        print("No positions loaded. Exiting.")
        return
    
    # Analyze positions
    print(f"Analyzing positions with baseline time {args.baseline_time}s and test times: {eval_times}")
    results = analyze_positions(
        positions, 
        eval_times, 
        args.baseline_time, 
        args.stockfish
    )
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print summary
    print("\nSummary of results:")
    print(f"{'Eval Time':10} | {'Time Ratio':10} | {'MAE':8} | {'RMSE':8} | {'Max Error':10} | {'Move Accuracy':12}")
    print("-" * 70)
    for eval_time in sorted(metrics["mae"].keys()):
        print(f"{eval_time:10.3f} | {metrics['eval_time_ratio'][eval_time]:10.3f} | {metrics['mae'][eval_time]:8.3f} | "
              f"{metrics['rmse'][eval_time]:8.3f} | {metrics['max_error'][eval_time]:10.3f} | "
              f"{metrics['move_accuracy'][eval_time]:12.1f}%")
    
    # Save results
    csv_output = output_dir / f"{args.output_prefix}.csv"
    save_results_to_csv(results, metrics, str(csv_output))
    
    # Generate and save plots
    plot_output = output_dir / f"{args.output_prefix}.png"
    plot_results(metrics, str(plot_output))


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")