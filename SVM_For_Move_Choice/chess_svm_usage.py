import chess
import chess.pgn
import chess.svg
import argparse
import os
from chess_svm_model import ChessMoveEvaluator
import sys
import io

def display_board_ascii(board):
    """Display the board in ASCII format"""
    print(board)

def evaluate_position(evaluator, fen=None):
    """
    Evaluate a position from FEN string or current position
    
    Args:
        evaluator (ChessMoveEvaluator): The trained model
        fen (str, optional): FEN string of position to analyze
    """
    if fen:
        try:
            board = chess.Board(fen)
        except ValueError as e:
            print(f"Invalid FEN: {e}")
            return
    else:
        board = chess.Board()  # Starting position
    
    display_board_ascii(board)
    
    # Evaluate all legal moves
    move_evaluations = []
    for move in board.legal_moves:
        confidence = evaluator.evaluate_move(board, move)
        move_evaluations.append((move, confidence))
    
    # Sort by confidence
    move_evaluations.sort(key=lambda x: x[1], reverse=True)
    
    # Print moves and their confidence scores
    print("\nMove evaluations (sorted by confidence):")
    print("------------------------------------------")
    print(f"{'Move':<10} {'Confidence':<10}")
    print("------------------------------------------")
    for move, confidence in move_evaluations:
        print(f"{board.san(move):<10} {confidence:.4f}")
    
    # Find and display the best move
    best_move, confidence = evaluator.find_best_move(board)
    print(f"\nBest move according to model: {board.san(best_move)} with confidence {confidence:.4f}")

def play_game(evaluator, fen=None):
    """
    Play a game against the SVM model
    
    Args:
        evaluator (ChessMoveEvaluator): The trained model
        fen (str, optional): Starting position in FEN
    """
    if fen:
        try:
            board = chess.Board(fen)
        except ValueError as e:
            print(f"Invalid FEN: {e}")
            return
    else:
        board = chess.Board()  # Starting position
    
    print("Play a game against the SVM model")
    print("Enter moves in UCI format (e.g., 'e2e4') or type 'quit' to exit")
    
    while not board.is_game_over():
        display_board_ascii(board)
        print(f"\nTurn: {'White' if board.turn == chess.WHITE else 'Black'}")
        
        if board.turn == chess.WHITE:  # Human plays White
            valid_move = False
            while not valid_move:
                move_uci = input("Your move (UCI format): ")
                if move_uci.lower() == 'quit':
                    return
                
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        valid_move = True
                        board.push(move)
                    else:
                        print("Illegal move")
                except ValueError:
                    print("Invalid move format. Use UCI notation (e.g., 'e2e4')")
        else:  # SVM model plays Black
            print("SVM model is thinking...")
            best_move, confidence = evaluator.find_best_move(board)
            print(f"SVM plays {board.san(best_move)} (confidence: {confidence:.4f})")
            board.push(best_move)
    
    # Game over
    display_board_ascii(board)
    print("\nGame over!")
    print(f"Result: {board.result()}")
    if board.is_checkmate():
        print("Checkmate!")
    elif board.is_stalemate():
        print("Stalemate!")
    elif board.is_insufficient_material():
        print("Insufficient material!")
    elif board.is_fifty_moves():
        print("Fifty-move rule!")
    elif board.is_repetition():
        print("Threefold repetition!")

def analyze_pgn(evaluator, pgn_file, position_count=5):
    """
    Analyze positions from a PGN file
    
    Args:
        evaluator (ChessMoveEvaluator): The trained model
        pgn_file (str): Path to PGN file
        position_count (int): Number of positions to analyze
    """
    with open(pgn_file) as pgn:
        game = chess.pgn.read_game(pgn)
        if game is None:
            print("Invalid PGN file or no games found")
            return
        
        # Print game info
        white = game.headers.get("White", "Unknown")
        black = game.headers.get("Black", "Unknown")
        result = game.headers.get("Result", "*")
        print(f"Game: {white} vs {black}, Result: {result}")
        
        # Sample positions evenly from the game
        moves = list(game.mainline_moves())
        total_moves = len(moves)
        
        if total_moves < position_count:
            position_count = total_moves
        
        # Sample positions evenly from the game
        step = max(1, total_moves // position_count)
        positions_to_analyze = list(range(0, total_moves, step))[:position_count]
        
        for pos_idx in positions_to_analyze:
            board = game.board()
            for i, move in enumerate(moves):
                if i == pos_idx:  # We've reached the position to analyze
                    break
                board.push(move)
            
            move_played = moves[pos_idx] if pos_idx < len(moves) else None
            
            print("\n" + "="*50)
            print(f"Position after move {pos_idx}:")
            display_board_ascii(board)
            
            # Evaluate all moves and rank them
            move_evaluations = []
            for move in board.legal_moves:
                confidence = evaluator.evaluate_move(board, move)
                is_actual = (move == move_played) if move_played else False
                move_evaluations.append((move, confidence, is_actual))
            
            # Sort by confidence
            move_evaluations.sort(key=lambda x: x[1], reverse=True)
            
            # Print top moves
            print("\nTop moves according to SVM model:")
            for i, (move, confidence, is_actual) in enumerate(move_evaluations[:5]):
                played_marker = " (played in game)" if is_actual else ""
                print(f"{i+1}. {board.san(move)}: {confidence:.4f}{played_marker}")
            
            # Find rank of the move actually played
            if move_played:
                actual_move_idx = next((i for i, (m, _, _) in enumerate(move_evaluations) if m == move_played), None)
                if actual_move_idx is not None:
                    print(f"\nMove played in game ({board.san(move_played)}) ranked #{actual_move_idx+1} out of {len(move_evaluations)}")

def batch_evaluate_pgn(evaluator, pgn_dir, output_file="evaluation_results.csv", num=5):
    """
    Batch evaluate multiple PGN files and collect statistics
    
    Args:
        evaluator (ChessMoveEvaluator): The trained model
        pgn_dir (str): Directory containing PGN files
        output_file (str): Output CSV file path
    """
    import pandas as pd
    
    results = []
    
    # Find PGN files
    pgn_files = []
    for file in os.listdir(pgn_dir):
        if file.endswith('.pgn'):
            pgn_files.append(os.path.join(pgn_dir, file))
    
    print(f"Found {len(pgn_files)} PGN files to evaluate")
    
    games_processed = 0
    for pgn_file in pgn_files:
        print(f"Processing {pgn_file}...")
        
        with open(pgn_file) as pgn:
            while True:
                print(f"Processing game {games_processed+1} of {num}")
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                if (games_processed+1) > num:
                    break

                
                
                # Extract game info
                white = game.headers.get("White", "Unknown")
                black = game.headers.get("Black", "Unknown")
                result = game.headers.get("Result", "*")
                
                moves = list(game.mainline_moves())
                board = game.board()
                
                move_ranks = []
                move_confidences = []
                
                # Analyze each position
                for move_idx, move_played in enumerate(moves):
                    # Evaluate all legal moves
                    move_evaluations = []
                    for move in board.legal_moves:
                        confidence = evaluator.evaluate_move(board, move)
                        move_evaluations.append((move, confidence))
                    
                    # Sort by confidence
                    move_evaluations.sort(key=lambda x: x[1], reverse=True)
                    
                    # Find rank of the move actually played
                    actual_move_rank = next((i+1 for i, (m, _) in enumerate(move_evaluations) if m == move_played), None)
                    actual_move_confidence = next((conf for m, conf in move_evaluations if m == move_played), None)
                    
                    if actual_move_rank is not None:
                        move_ranks.append(actual_move_rank)
                        move_confidences.append(actual_move_confidence)
                    
                    # Make the move and continue
                    board.push(move_played)
                
                # Compute statistics
                top1_accuracy = sum(1 for r in move_ranks if r == 1) / len(move_ranks) if move_ranks else 0
                top3_accuracy = sum(1 for r in move_ranks if r <= 3) / len(move_ranks) if move_ranks else 0
                top5_accuracy = sum(1 for r in move_ranks if r <= 5) / len(move_ranks) if move_ranks else 0
                
                avg_rank = sum(move_ranks) / len(move_ranks) if move_ranks else None
                median_rank = sorted(move_ranks)[len(move_ranks)//2] if move_ranks else None
                avg_confidence = sum(move_confidences) / len(move_confidences) if move_confidences else None
                
                # Store game results
                results.append({
                    'pgn_file': os.path.basename(pgn_file),
                    'white': white,
                    'black': black,
                    'result': result,
                    'moves': len(moves),
                    'top1_accuracy': top1_accuracy,
                    'top3_accuracy': top3_accuracy,
                    'top5_accuracy': top5_accuracy,
                    'avg_rank': avg_rank,
                    'median_rank': median_rank,
                    'avg_confidence': avg_confidence
                })

                games_processed += 1
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Evaluation results saved to {output_file}")
        
        # Print summary statistics
        print("\nOverall Statistics:")
        print(f"Top-1 Accuracy: {df['top1_accuracy'].mean():.4f}")
        print(f"Top-3 Accuracy: {df['top3_accuracy'].mean():.4f}")
        print(f"Top-5 Accuracy: {df['top5_accuracy'].mean():.4f}")
        print(f"Average Move Rank: {df['avg_rank'].mean():.2f}")
        print(f"Average Confidence: {df['avg_confidence'].mean():.4f}")
    else:
        print("No results collected")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Chess SVM Model Usage")
    parser.add_argument("--model", type=str, default="chess_svm_model.pkl",
                      help="Path to the trained model file")
    parser.add_argument("--stockfish", type=str, default="stockfish",
                      help="Path to the Stockfish engine executable")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Evaluate position command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a chess position")
    evaluate_parser.add_argument("--fen", type=str, default=None,
                               help="FEN string of the position to evaluate")
    
    # Play game command
    play_parser = subparsers.add_parser("play", help="Play a game against the SVM model")
    play_parser.add_argument("--fen", type=str, default=None,
                           help="FEN string of the starting position")
    
    # Analyze PGN command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze positions from a PGN file")
    analyze_parser.add_argument("--pgn", type=str, required=True,
                              help="Path to the PGN file to analyze")
    analyze_parser.add_argument("--positions", type=int, default=5,
                              help="Number of positions to analyze")
    
    # Batch evaluate command
    batch_parser = subparsers.add_parser("batch", help="Batch evaluate multiple PGN files")
    batch_parser.add_argument("--pgn_dir", type=str, required=True,
                            help="Directory containing PGN files")
    batch_parser.add_argument("--num", type=int, default = 5,
                              help="Number of PGN games to analyse from files")
    batch_parser.add_argument("--output", type=str, default="evaluation_results.csv",
                            help="Output CSV file path")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Initialize the model
    evaluator = ChessMoveEvaluator(stockfish_path=args.stockfish)
    
    try:
        # Load the trained model
        evaluator.load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process command
    if args.command == "evaluate":
        evaluate_position(evaluator, args.fen)
    elif args.command == "play":
        play_game(evaluator, args.fen)
    elif args.command == "analyze":
        analyze_pgn(evaluator, args.pgn, args.positions)
    elif args.command == "batch":
        batch_evaluate_pgn(evaluator, args.pgn_dir, args.output, args.num)
    else:
        print("No command specified. Use --help for usage information.")

if __name__ == "__main__":
    main()