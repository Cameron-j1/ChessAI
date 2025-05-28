import chess
import chess.pgn
import chess.engine
from pathlib import Path
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_stockfish():
    """Find Stockfish executable in common locations."""
    common_locations = [
        "stockfish",  # If in PATH
        "/usr/local/bin/stockfish",
        "/usr/games/stockfish",
        "/usr/bin/stockfish"
    ]
    
    for location in common_locations:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(location)
            engine.quit()
            return location
        except Exception:
            continue
    
    raise FileNotFoundError("Stockfish not found. Please install it using your package manager.")

def complete_game_with_stockfish(board, engine, time_limit):
    """Complete a chess game using Stockfish for both sides."""
    moves = []
    while not board.is_game_over():
        result = engine.play(board, chess.engine.Limit(time=time_limit))
        moves.append(result.move)
        board.push(result.move)
    return board, moves

def process_pgn_file(input_pgn_path, output_pgn_path, stockfish_path=None, time_limit=0.1, log_frequency=100):
    """Process a PGN file and complete non-checkmate games with Stockfish."""
    if stockfish_path is None:
        stockfish_path = find_stockfish()
    
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    try:
        input_pgn = open(input_pgn_path)
        output_pgn = open(output_pgn_path, 'w', encoding='utf-8')
        
        games_processed = 0
        games_completed = 0
        
        while True:
            try:
                game = chess.pgn.read_game(input_pgn)
                if game is None:
                    break
                
                games_processed += 1
                if games_processed % log_frequency == 0:
                    logger.info(f"Processed {games_processed} games...")
                
                # Create a new game with the same headers
                new_game = chess.pgn.Game()
                for key in game.headers:
                    new_game.headers[key] = game.headers[key]
                
                # Get to the end position and copy all original moves
                board = chess.Board()
                node = new_game
                
                for move in game.mainline_moves():
                    board.push(move)
                    node = node.add_variation(move)
                
                # If game didn't end in checkmate, complete it with Stockfish
                if not board.is_checkmate():
                    games_completed += 1
                    logger.info(f"Completing game {games_processed} with Stockfish...")
                    
                    # Complete the game with Stockfish and get the moves
                    final_board, stockfish_moves = complete_game_with_stockfish(board, engine, time_limit)
                    
                    # Add Stockfish moves to the main line
                    for move in stockfish_moves:
                        node = node.add_variation(move)
                    
                    # Update the result header based on the final position
                    if final_board.is_checkmate():
                        new_game.headers["Result"] = "1-0" if final_board.turn == chess.BLACK else "0-1"
                    elif final_board.is_stalemate() or final_board.is_insufficient_material():
                        new_game.headers["Result"] = "1/2-1/2"
                
                print(new_game, file=output_pgn, end="\n\n")
                
            except Exception as e:
                logger.error(f"Error processing game: {e}")
                continue
        
        logger.info(f"Processing complete!")
        logger.info(f"Total games processed: {games_processed}")
        logger.info(f"Games completed by Stockfish: {games_completed}")
        
    finally:
        engine.quit()
        input_pgn.close()
        output_pgn.close()

if __name__ == "__main__":
    # Configuration parameters
    STOCKFISH_PATH = '/usr/games/stockfish'  # Path to Stockfish executable
    STOCKFISH_TIME_LIMIT = 0.03  # Time limit per move in seconds
    LOG_FREQUENCY = 100  # How often to log progress (every N games)
    
    input_pgn = 'standardover2000-2021.pgn'
    output_pgn = 'standardover2000-2021_processed.pgn'
    
    if not Path(input_pgn).exists():
        print(f"Input file {input_pgn} does not exist!")
        sys.exit(1)
    
    process_pgn_file(
        input_pgn_path=input_pgn,
        output_pgn_path=output_pgn,
        stockfish_path=STOCKFISH_PATH,
        time_limit=STOCKFISH_TIME_LIMIT,
        log_frequency=LOG_FREQUENCY
    )
