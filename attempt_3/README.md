# ChessAI - Deep Learning Chess Engine

A PyTorch-based chess AI that uses deep learning to predict moves from board positions. The project supports multiple model architectures including spatial CNN models and attention-enhanced models.

## Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- Stockfish chess engine

### Virtual Environment Setup

1. Create a virtual environment:
```bash
python -m venv chess_ai_env
```

2. Activate the virtual environment:
```bash
# On Linux/Mac:
source chess_ai_env/bin/activate

# On Windows:
chess_ai_env\Scripts\activate
```

3. Install system dependencies:
```bash
sudo apt-get install stockfish
```

4. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Data Requirements and Processing

### PGN Data Structure

The project expects PGN (Portable Game Notation) files to be placed in the root directory alongside the training scripts. The system will automatically process all `.pgn` files found in the current directory.

### Processing PGN Files

Before training, PGN files need to be processed to ensure all games have proper endings:

```bash
python pgn_process.py
```

**What `pgn_process.py` does:**
- Reads PGN files from the current directory
- Identifies games that didn't end in checkmate
- Uses Stockfish to complete these games to checkmate/draw
- Saves processed PGN files with `_processed.pgn` suffix
- Moves completed games to the appropriate location

**Configuration options in `pgn_process.py`:**
- `STOCKFISH_PATH`: Path to Stockfish executable (default: `/usr/games/stockfish`)
- `STOCKFISH_TIME_LIMIT`: Time limit per move in seconds (default: 0.03)
- `LOG_FREQUENCY`: Progress logging frequency (default: every 100 games)

### Data Storage

Processed training data is stored in the `npz/` directory:
- Raw PGN files → Processed board positions and moves
- Saved as `.npz` files containing NumPy arrays
- Training scripts automatically detect and load existing `.npz` files
- If no `.npz` files exist, scripts will process PGN files on-the-fly

## Training Models

### 1. Spatial Attention Model (`train_spatial_attention.py`)

Trains an attention-enhanced CNN model with spatial move representation:

```bash
python train_spatial_attention.py
```

**Features:**
- Uses `ChessModelAttention` architecture with 8 transformer blocks
- Spatial move encoding (8x8 grid for from/to squares)
- Custom `SpatialMoveLoss` function
- Batch size: 256 (reduced for larger model)
- Learning rate: 0.0005 with CosineAnnealingLR scheduler
- Weight decay: 2e-4 for regularization
- Dropout rate: 0.2

**Output:** Saves model checkpoints in `models/` as `attention_model_checkpoint_epoch_X.pth`

### 2. Spatial CNN Model (`train_spatial.py`)

Trains a spatial CNN model using the `ChessModelSpatial` architecture:

```bash
python train_spatial.py
```

**Features:**
- Uses `ChessModelSpatial` from `chess_model_class.py`
- Spatial move representation (8x8 grid)
- Batch size: 512
- Learning rate: 0.001 with CosineAnnealingLR scheduler
- Weight decay: 1e-4
- Dropout rate: 0.2

**Output:** Saves model checkpoints in `models/` as `spatial_model_checkpoint_epoch_X.pth`

### 3. Alternative Training (`train.py`)

Basic training script for categorical move prediction:

```bash
python train.py
```

**Features:**
- Uses traditional categorical move encoding
- Requires move mapping files for encoding/decoding
- Simpler architecture but larger output vocabulary

### Data Requirements for Training

- **Minimum:** Several PGN files in the root directory
- **Recommended:** PGN files with (rating > 2000)
- **RAM Constraints:** Training scripts limit dataset to match RAM constraints, guess and check how much you can load but i found 3.9M positions was the limit for 32gb of ram
- **Storage:** Processed data files can up to 10GB each in `npz/` directory

### Training Process

1. **Data Loading:**
   - Scripts first check for existing `.npz` files in `npz/` directory
   - If found, loads preprocessed data directly
   - If not found, processes PGN files from root directory

2. **Data Processing:**
   - Converts games to board representations and move targets
   - Uses spatial encoding for CNN models
   - Saves processed data to `npz/` for future use

3. **Model Training:**
   - Automatic GPU detection and usage
   - Regular checkpoint saving every few epochs
   - Progress monitoring with loss tracking
   - Cosine annealing learning rate scheduling

## Running and Evaluating Models

### Model Evaluation (`evaluate.py`)

Comprehensive evaluation script supporting multiple model types:

```bash
python evaluate.py
```

**Supported Model Types:**
- `categorical`: Traditional move classification models
- `spatial`: CNN models with spatial move representation  
- `attention`: Attention-enhanced CNN models

**Features:**
- Model vs Model gameplay
- Model vs Stockfish evaluation
- Checkmate threat detection
- Piece safety analysis
- Move validation and filtering
- Performance statistics

**Configuration:**
- Edit the `main()` function in `evaluate.py` to set:
  - Model paths and types
  - Number of games to play
  - Time controls
  - Evaluation modes

### Human vs AI Gameplay (`play_human.py`)

Interactive GUI for playing against the AI:

```bash
python play_human.py
```

**Features:**
- Graphical chess board using tkinter and chess.svg
- Human plays as White, AI plays as Black
- Move input in UCI format (e.g., "e2e4")
- Visual board updates with last move highlighting
- Check/checkmate detection and display
- New game functionality

**Configuration:**
- Modify model path in `ChessGUI.__init__()`:
```python
model_path = 'models/spatial_model_checkpoint_epoch_15.pth'
model_type = 'spatial'  # or 'attention' or 'categorical'
```

### Model Files Location

Trained models are stored in the `models/` directory:
- `attention_model_checkpoint_epoch_X.pth`: Attention models
- `spatial_model_checkpoint_epoch_X.pth`: Spatial CNN models  
- `spatial_model_dropout_checkpoint_epoch_X.pth`: Spatial models with dropout
- Various legacy models for compatibility

### Running Evaluations

1. **Quick Model Test:**
   ```bash
   python evaluate.py
   # Edit main() function to configure evaluation
   ```

2. **Human Gameplay:**
   ```bash
   python play_human.py
   # Uses model specified in the script
   ```

3. **Model Architecture Visualization:**
   ```bash
   python visualize_model.py        # Basic visualization
   python visualize_model_simple.py # Detailed visualization
   ```

## Project Structure

```
├── README.md                     # This documentation
├── requirements.txt              # Python dependencies
├── pgn/                         # PGN files directory
│   ├── used/                    # Processed PGN files
│   └── *.pgn                    # Raw PGN files
├── npz/                         # Processed training data
│   └── spatial_training_data_*.npz
├── models/                      # Trained model checkpoints
├── chess_model_class*.py        # Model architecture definitions
├── train*.py                    # Training scripts
├── evaluate.py                  # Model evaluation and testing
├── play_human.py               # Human vs AI interface
├── pgn_process.py              # PGN file processing
├── other_functions.py          # Utility functions
├── dataset_class.py            # PyTorch dataset classes
└── visualize_model*.py         # Model visualization tools
```

## Tips and Best Practices

1. **Training:** Start with spatial attention models as they perform best
2. **Data:** Use high-quality games (rating > 2000) for better training results
3. **Memory:** Monitor RAM usage during training; reduce positions loaded if needed
4. **Evaluation:** Test models against multiple opponents for robust evaluation
5. **Checkpoints:** Keep multiple epoch checkpoints as later epochs may overfit

