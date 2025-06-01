import os
import numpy as np # type: ignore
from chess import pgn # type: ignore
from tqdm import tqdm # type: ignore
from pathlib import Path

from other_functions import create_input_for_nn, encode_moves_spatial

def load_single_pgn(file_path):
    """Load a single PGN file and return list of games"""
    games = []
    with open(file_path, 'r') as pgn_file:
        while True:
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games

def save_chunk(X, y_spatial, output_dir, file_index, chunk_index):
    """Save a chunk of data"""
    output_path = output_dir / f"data_{file_index}_{chunk_index}"
    np.savez_compressed(
        output_path,
        X=X,
        y=y_spatial
    )
    return len(X)

def process_pgn_file(pgn_path, output_dir, file_index):
    """Process a single PGN file and save its data in chunks"""
    print(f"\nProcessing {pgn_path.name}")
    
    # Load games from PGN
    games = load_single_pgn(pgn_path)
    print(f"Loaded {len(games)} games")
    
    # Create neural network input
    X, y = create_input_for_nn(games)
    
    # Convert moves to spatial encoding
    y_spatial = encode_moves_spatial(y)
    
    # Split into chunks of approximately 100,000 positions
    CHUNK_SIZE = 100000
    total_positions = 0
    chunks_info = []
    
    for chunk_idx in range(0, len(X), CHUNK_SIZE):
        X_chunk = X[chunk_idx:chunk_idx + CHUNK_SIZE]
        y_chunk = y_spatial[chunk_idx:chunk_idx + CHUNK_SIZE]
        
        # Save chunk
        chunk_name = f"data_{file_index}_{len(chunks_info)}.npz"
        positions = save_chunk(X_chunk, y_chunk, output_dir, file_index, len(chunks_info))
        total_positions += positions
        chunks_info.append({
            'filename': chunk_name,
            'size': positions
        })
        
        print(f"Saved chunk {len(chunks_info)} with {positions} positions")
    
    # Free memory
    del X, y, y_spatial, games
    
    print(f"Saved total of {total_positions} positions in {len(chunks_info)} chunks")
    return total_positions, chunks_info

def main():
    # Create necessary directories
    pgn_dir = Path("pgn")
    output_dir = Path("processed_data")
    output_dir.mkdir(exist_ok=True)
    
    if not pgn_dir.exists():
        print("Creating pgn directory...")
        pgn_dir.mkdir()
        print("Please place your PGN files in the 'pgn' directory and run this script again.")
        return
    
    # Get list of PGN files
    pgn_files = list(pgn_dir.glob("*.pgn"))
    
    if not pgn_files:
        print("No PGN files found in the 'pgn' directory!")
        return
    
    print(f"Found {len(pgn_files)} PGN files")
    
    # Process each PGN file
    total_positions = 0
    all_chunks = []
    
    for idx, pgn_path in enumerate(pgn_files):
        try:
            positions, chunks_info = process_pgn_file(pgn_path, output_dir, idx)
            total_positions += positions
            all_chunks.extend(chunks_info)
        except Exception as e:
            print(f"Error processing {pgn_path.name}: {str(e)}")
            continue
    
    # Save metadata
    metadata = {
        "num_files": len(pgn_files),
        "total_positions": total_positions,
        "chunks": all_chunks,
        "target_chunk_size": 100000
    }
    
    np.save(output_dir / "metadata.npy", metadata)
    print(f"\nProcessing complete! Total positions: {total_positions}")
    print(f"Data files saved in {output_dir} ({len(all_chunks)} chunks)")

if __name__ == "__main__":
    main() 