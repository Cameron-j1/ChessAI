import os
import argparse
import chess
import chess.pgn
import numpy as np
import pandas as pd
import joblib
from chess_svm_model import ChessMoveEvaluator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Chess Move SVM Trainer")
    parser.add_argument("--pgn_dir", type=str, default="pgn_files", 
                      help="Directory containing PGN files")
    parser.add_argument("--stockfish_path", type=str, default="stockfish", 
                      help="Path to Stockfish executable")
    parser.add_argument("--num_games", type=int, default=50, 
                      help="Number of games to process")
    parser.add_argument("--model_path", type=str, default="chess_svm_model.pkl", 
                      help="Path to save/load the model")
    parser.add_argument("--mode", choices=["train", "test", "analyze"], default="train",
                      help="Mode of operation: train, test, or analyze")
    parser.add_argument("--visualize", action="store_true", 
                      help="Visualize training results")
    return parser.parse_args()

def collect_pgn_files(pgn_dir):
    """Collect all PGN files in the specified directory"""
    pgn_files = []
    for file in os.listdir(pgn_dir):
        if file.endswith(".pgn"):
            pgn_files.append(os.path.join(pgn_dir, file))
    return pgn_files

def visualize_results(model, X_test, y_test):
    """Visualize model performance"""
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted Label')
    axes[0, 0].set_ylabel('True Label')
    
    # Probability Distribution
    axes[0, 1].hist(y_prob[y_test == 0], bins=20, alpha=0.5, label='Class 0')
    axes[0, 1].hist(y_prob[y_test == 1], bins=20, alpha=0.5, label='Class 1')
    axes[0, 1].set_title('Probability Distribution')
    axes[0, 1].set_xlabel('Probability of Class 1')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend()
    
    # Feature Importance (for linear kernel only)
    if hasattr(model['svm'], 'coef_'):
        feature_importance = model['svm'].coef_[0]
        sorted_idx = np.argsort(np.abs(feature_importance))[::-1][:20]  # Top 20 features
        axes[1, 0].barh(range(20), feature_importance[sorted_idx])
        axes[1, 0].set_title('Top 20 Feature Importance')
        axes[1, 0].set_xlabel('Importance')
        axes[1, 0].set_yticks(range(20))
        axes[1, 0].set_yticklabels([f'Feature {i}' for i in sorted_idx])
    else:
        axes[1, 0].text(0.5, 0.5, 'Feature importance not available for non-linear kernels', 
                       ha='center', va='center')
    
    # ROC Curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    axes[1, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    axes[1, 1].plot([0, 1], [0, 1], 'k--')
    axes[1, 1].set_title('ROC Curve')
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('model_performance.png')
    plt.close()
    print("Visualization saved to 'model_performance.png'")

def analyze_positions(evaluator, pgn_file, num_positions=10):
    """Analyze specific positions from a game"""
    results = []
    
    with open(pgn_file) as pgn:
        game = chess.pgn.read_game(pgn)
        if game is None:
            print("No game found in the PGN file")
            return
        
        # Extract player names and game info
        white = game.headers.get("White", "Unknown")
        black = game.headers.get("Black", "Unknown")
        date = game.headers.get("Date", "Unknown")
        
        print(f"Analyzing {white} vs {black}, {date}")
        
        # Analyze every few positions
        board = game.board()
        moves = list(game.mainline_moves())
        positions_to_analyze = np.linspace(0, len(moves)-1, num_positions, dtype=int)
        
        for i, move_idx in enumerate(positions_to_analyze):
            # Reset board to starting position
            board = game.board()
            
            # Play moves up to the position to analyze
            for j in range(move_idx):
                board.push(moves[j])
            
            print(f"\nPosition {i+1}/{num_positions} (after move {move_idx}):")
            print(board)
            
            # Get the actual move played in the game
            actual_move = moves[move_idx] if move_idx < len(moves) else None
            
            # Evaluate all legal moves
            move_evaluations = []
            for move in board.legal_moves:
                confidence = evaluator.evaluate_move(board, move)
                move_evaluations.append((move, confidence))
            
            # Sort by confidence
            move_evaluations.sort(key=lambda x: x[1], reverse=True)
            
            # Print top 5 moves
            print("Top 5 moves according to SVM model:")
            for j, (move, confidence) in enumerate(move_evaluations[:5]):
                is_actual = actual_move == move if actual_move else False
                print(f"{j+1}. {board.san(move)}: {confidence:.4f}" + (" (played in game)" if is_actual else ""))
            
            # Find rank of actual move
            if actual_move:
                actual_move_rank = next((i+1 for i, (move, _) in enumerate(move_evaluations) if move == actual_move), None)
                if actual_move_rank:
                    print(f"Actual move {board.san(actual_move)} ranked #{actual_move_rank} out of {len(move_evaluations)}")
                    results.append(actual_move_rank)
                else:
                    print("Could not find the rank of the actual move")
    
    # Calculate statistics on results
    if results:
        print("\nStatistics on model performance:")
        print(f"Average rank of actual move: {np.mean(results):.2f}")
        print(f"Median rank of actual move: {np.median(results):.2f}")
        print(f"Top-1 accuracy: {sum(1 for r in results if r == 1) / len(results):.2f}")
        print(f"Top-3 accuracy: {sum(1 for r in results if r <= 3) / len(results):.2f}")
        print(f"Top-5 accuracy: {sum(1 for r in results if r <= 5) / len(results):.2f}")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Initialize evaluator
    evaluator = ChessMoveEvaluator(stockfish_path=args.stockfish_path)
    
    if args.mode == "train":
        # Collect PGN files
        pgn_files = collect_pgn_files(args.pgn_dir)
        if not pgn_files:
            print(f"No PGN files found in {args.pgn_dir}")
            return
        
        print(f"Found {len(pgn_files)} PGN files")
        
        # Prepare training data
        X, y = evaluator.prepare_training_data(pgn_files, num_games=args.num_games)
        
        # Train model
        accuracy = evaluator.train(X, y)
        print(f"Model trained with validation accuracy: {accuracy:.4f}")
        
        # Save model
        evaluator.save_model(args.model_path)
        
        # Optionally visualize results
        if args.visualize:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            visualize_results(evaluator.model, X_test, y_test)
            
    elif args.mode == "test":
        # Load the model
        evaluator.load_model(args.model_path)
        
        # Create a test board position
        board = chess.Board()
        
        # Evaluate each legal move
        print("Evaluating opening position moves:")
        for move in board.legal_moves:
            confidence = evaluator.evaluate_move(board, move)
            print(f"{board.san(move)}: {confidence:.4f}")
        
        # Find the best move
        best_move, confidence = evaluator.find_best_move(board)
        print(f"\nBest move according to model: {board.san(best_move)} with confidence {confidence:.4f}")
        
    elif args.mode == "analyze":
        # Load the model
        evaluator.load_model(args.model_path)
        
        # Find a PGN file to analyze
        pgn_files = collect_pgn_files(args.pgn_dir)
        if not pgn_files:
            print(f"No PGN files found in {args.pgn_dir}")
            return
            
        analyze_positions(evaluator, pgn_files[0])

if __name__ == "__main__":
    main()