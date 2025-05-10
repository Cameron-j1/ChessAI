# Chess Move Evaluator with SVM

This project creates a Support Vector Machine (SVM) model for evaluating chess moves. It analyzes chess games in PGN format, extracts board positions and moves, evaluates them with Stockfish, and uses this data to train an SVM model that predicts the quality of candidate moves.

## Features

- Parse and extract features from chess games in PGN format
- Extract board and move features for machine learning
- Train an SVM model to predict move quality
- Evaluate board positions and suggest best moves
- Interactive chess game interface to play against the model
- Comprehensive analysis of chess positions and game statistics

## Requirements

- Python 3.7+
- Required packages:
  - python-chess
  - scikit-learn
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - joblib
  - tqdm
  - Stockfish chess engine

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/chess-svm-evaluator.git
   cd chess-svm-evaluator
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download and install Stockfish:
   - Download from [Stockfish's official website](https://stockfishchess.org/download/)
   - Make sure the executable is in your PATH or specify its location with the `--stockfish` parameter

## Project Structure

- `chess_svm_model.py`: Core model implementation with feature extraction and SVM training
- `chess_svm_pipeline.py`: Complete pipeline for training and analyzing the model
- `chess_svm_usage.py`: Command-line tools for using the trained model

## Usage

### Training the Model

```bash
python chess_svm_pipeline.py --pgn_dir ./pgn_files --stockfish_path /path/to/stockfish --num_games 50 --mode train
```

### Testing the Model

```bash
python chess_svm_pipeline.py --pgn_dir ./pgn_files --stockfish_path /path/to/stockfish --mode test
```

### Analyzing Chess Positions

```bash
python chess_svm_usage.py --model chess_svm_model.pkl --stockfish /path/to/stockfish evaluate --fen "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
```

### Playing Against the Model

```bash
python chess_svm_usage.py --model chess_svm_model.pkl --stockfish /path/to/stockfish play
```

### Analyzing PGN Games

```bash
python chess_svm_usage.py --model chess_svm_model.pkl --stockfish /path/to/stockfish analyze --pgn game.pgn --positions 10
```

### Batch Evaluation

```bash
python chess_svm_usage.py --model chess_svm_model.pkl --stockfish /path/to/stockfish batch --pgn_dir ./pgn_files
```

## Model Features

The model extracts numerous features from chess positions and candidate moves:

### Board Features
- Material balance
- Piece placement
- Center control
- Mobility (legal move count)
- King safety
- Pawn structure
- Game phase

### Move Features
- Capture value
- Check giving potential
- Moving to attacked squares
- Center control
- Piece type being moved
- Promotion potential
- Castling

## How It Works

1. **Data Preparation**:
   - Parse PGN files to extract board positions and moves
   - Use Stockfish to evaluate positions before and after moves
   - Label moves as "good" (1) or "bad" (0) based on evaluation changes

2. **Feature Extraction**:
   - Extract meaningful features from board positions
   - Extract features specific to candidate moves
   - Combine features into a comprehensive feature vector

3. **Model Training**:
   - Train an SVM model using scikit-learn
   - Use standardization for feature scaling
   - Tune hyperparameters for optimal performance

4. **Prediction**:
   - For each legal move in a position, compute features
   - Get confidence scores from the SVM model
   - Rank moves by confidence

## Extending the Model

To improve the model:

1. **Add more features**:
   - Update `_extract_board_features()` and `_extract_move_features()` in `chess_svm_model.py`
   
2. **Try different ML algorithms**:
   - Replace SVC with other algorithms in the `train()` method

3. **Enhance training data**:
   - Use more games or higher quality games
   - Increase Stockfish analysis depth for better evaluations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits

- [python-chess](https://python-chess.readthedocs.io/) for chess implementation
- [scikit-learn](https://scikit-learn.org/) for machine learning tools
- [Stockfish](https://stockfishchess.org/) for chess position evaluation