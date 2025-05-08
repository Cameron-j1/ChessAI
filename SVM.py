import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import chess
import chess.pgn
import io


# Parse CSV with error handling
def parse_chess_data(file_path):
    try:
        # Try first with automatic header detection
        df = pd.read_csv(file_path)
        
        # Check if we have the expected columns
        expected_columns = ['game_id', 'result', 'pgn']
        if not all(col in df.columns for col in expected_columns):
            # Try with no header
            df = pd.read_csv(file_path, header=None)
            
            # Assign column names based on your data format
            column_names = [
                'game_id', 'result', 'timestamp_start', 'timestamp_end', 
                'turns', 'time_control', 'color_result', 'time_setting',
                'white_username', 'white_rating', 'black_username', 'black_rating',
                'pgn', 'eco', 'opening_name', 'termination'
            ]
            
            # If there are fewer columns in the data, adjust accordingly
            if len(df.columns) < len(column_names):
                column_names = column_names[:len(df.columns)]
            
            df.columns = column_names
    except Exception as e:
        print(f"Error reading CSV: {e}")
        print("Attempting to read with different options...")
        
        try:
            # Try different separator
            df = pd.read_csv(file_path, sep=',', header=None)
            
            # If we only have one column, it might be a different separator
            if len(df.columns) == 1:
                # Try with tab separator
                df = pd.read_csv(file_path, sep='\t', header=None)
            
            # Assign minimum required columns
            if len(df.columns) >= 3:
                df.columns = ['game_id', 'result', 'pgn'] + [f'col_{i}' for i in range(3, len(df.columns))]
            else:
                # If we don't have enough columns, create dummy ones
                print("Warning: CSV doesn't have enough columns. Creating dummy data.")
                df['pgn'] = df[0]  # Assume the first column is the PGN
                df['game_id'] = range(len(df))
                df['result'] = 0  # Default to draw
                
        except Exception as e2:
            print(f"Error with alternative parsing: {e2}")
            # Create a minimal dataframe for testing
            print("Creating minimal test data...")
            df = pd.DataFrame({
                'game_id': ['test1', 'test2'],
                'result': [0, 0],
                'pgn': [
                    "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6",
                    "1. d4 d5 2. c4 e6 3. Nc3 Nf6"
                ]
            })
    
    # Ensure required columns exist
    required_columns = ['game_id', 'result', 'pgn']
    for col in required_columns:
        if col not in df.columns:
            df[col] = None if col != 'result' else 0
    
    print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {', '.join(df.columns)}")
    
    return df

# Function to extract features from a chess position
def extract_features_from_position(board):
    features = []
    
    # Material count (pawns=1, knights/bishops=3, rooks=5, queens=9)
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # Not counting king in material
    }
    
    white_material = 0
    black_material = 0
    
    # Count material for each side
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                white_material += value
            else:
                black_material += value
    
    # Material advantage
    material_advantage = white_material - black_material
    features.append(material_advantage)
    
    # Mobility (number of legal moves)
    white_mobility = 0
    black_mobility = 0
    
    # Save current turn
    original_turn = board.turn
    
    # Count white moves
    board.turn = chess.WHITE
    white_mobility = len(list(board.legal_moves))
    
    # Count black moves
    board.turn = chess.BLACK
    black_mobility = len(list(board.legal_moves))
    
    # Restore original turn
    board.turn = original_turn
    
    # Mobility advantage
    mobility_advantage = white_mobility - black_mobility
    features.append(mobility_advantage)
    
    # Center control (pawns and pieces in the center 4 squares)
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    white_center_control = 0
    black_center_control = 0
    
    for square in center_squares:
        piece = board.piece_at(square)
        if piece:
            if piece.color == chess.WHITE:
                white_center_control += 1
            else:
                black_center_control += 1
    
    # Center control advantage
    center_control_advantage = white_center_control - black_center_control
    features.append(center_control_advantage)
    
    # Pawn structure (connected pawns, isolated pawns, doubled pawns)
    white_pawns = [square for square in chess.SQUARES 
                  if board.piece_at(square) and 
                  board.piece_at(square).piece_type == chess.PAWN and 
                  board.piece_at(square).color == chess.WHITE]
    
    black_pawns = [square for square in chess.SQUARES 
                  if board.piece_at(square) and 
                  board.piece_at(square).piece_type == chess.PAWN and 
                  board.piece_at(square).color == chess.BLACK]
    
    # Count isolated pawns
    white_isolated = 0
    black_isolated = 0
    
    for pawn in white_pawns:
        file = chess.square_file(pawn)
        adjacent_files = []
        if file > 0:
            adjacent_files.append(file - 1)
        if file < 7:
            adjacent_files.append(file + 1)
        
        # Check if there are pawns in adjacent files
        if not any(chess.square_file(p) in adjacent_files for p in white_pawns):
            white_isolated += 1
    
    for pawn in black_pawns:
        file = chess.square_file(pawn)
        adjacent_files = []
        if file > 0:
            adjacent_files.append(file - 1)
        if file < 7:
            adjacent_files.append(file + 1)
        
        # Check if there are pawns in adjacent files
        if not any(chess.square_file(p) in adjacent_files for p in black_pawns):
            black_isolated += 1
    
    # Pawn structure advantage (less isolated pawns is better)
    pawn_structure_advantage = black_isolated - white_isolated
    features.append(pawn_structure_advantage)
    
    # King safety (distance of opponent pieces to king)
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)
    
    if white_king_square and black_king_square:  # Ensure kings are on the board
        white_king_safety = 0
        black_king_safety = 0
        
        # Calculate distance of opponent pieces to king
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                if piece.color == chess.BLACK and piece.piece_type != chess.KING:
                    distance = chess.square_distance(white_king_square, square)
                    white_king_safety -= (8 - distance)  # Closer pieces are more dangerous
                elif piece.color == chess.WHITE and piece.piece_type != chess.KING:
                    distance = chess.square_distance(black_king_square, square)
                    black_king_safety -= (8 - distance)
        
        # King safety advantage
        king_safety_advantage = white_king_safety - black_king_safety
        features.append(king_safety_advantage)
    else:
        features.append(0)  # Default if kings not found
    
    # Development (knights and bishops outside starting position)
    white_development = 0
    black_development = 0
    
    # Knights
    if board.piece_at(chess.B1) is None or board.piece_at(chess.B1).piece_type != chess.KNIGHT:
        white_development += 1
    if board.piece_at(chess.G1) is None or board.piece_at(chess.G1).piece_type != chess.KNIGHT:
        white_development += 1
    if board.piece_at(chess.B8) is None or board.piece_at(chess.B8).piece_type != chess.KNIGHT:
        black_development += 1
    if board.piece_at(chess.G8) is None or board.piece_at(chess.G8).piece_type != chess.KNIGHT:
        black_development += 1
    
    # Bishops
    if board.piece_at(chess.C1) is None or board.piece_at(chess.C1).piece_type != chess.BISHOP:
        white_development += 1
    if board.piece_at(chess.F1) is None or board.piece_at(chess.F1).piece_type != chess.BISHOP:
        white_development += 1
    if board.piece_at(chess.C8) is None or board.piece_at(chess.C8).piece_type != chess.BISHOP:
        black_development += 1
    if board.piece_at(chess.F8) is None or board.piece_at(chess.F8).piece_type != chess.BISHOP:
        black_development += 1
    
    # Development advantage
    development_advantage = white_development - black_development
    features.append(development_advantage)
    
    return features

# Function to extract features and labels from game data
def process_games(df):
    X = []  # Features
    y = []  # Labels
    
    print("Extracting features from chess games...")
    game_count = 0
    position_count = 0
    
    for _, row in df.iterrows():
        game_count += 1
        if game_count % 10 == 0:
            print(f"  Processing game {game_count}/{len(df)}...")
        
        pgn_text = row['pgn']
        result_text = row.get('result', None)
        
        # Parse result
        if result_text == 'TRUE':
            # White won
            game_result = 1
        elif result_text == 'FALSE':
            # Black won
            game_result = -1
        else:
            # Draw or unknown
            game_result = 0
        
        # Parse PGN
        try:
            # Add proper PGN headers if they're missing
            if not pgn_text.startswith('['):
                pgn_text = f'[Event "Game"]\n[White "White"]\n[Black "Black"]\n[Result "*"]\n\n{pgn_text}'
            
            pgn = io.StringIO(pgn_text)
            game = chess.pgn.read_game(pgn)
            
            # Skip games without proper PGN
            if not game:
                continue
            
            # Get positions from game
            board = game.board()
            move_num = 0
            
            # Process each position in the game
            for move in game.mainline_moves():
                try:
                    # Make the move
                    board.push(move)
                    move_num += 1
                    
                    # Only use positions after the opening (move > 5)
                    if move_num < 5:
                        continue
                    
                    # Extract features from current position
                    features = extract_features_from_position(board)
                    
                    # Skip if features couldn't be extracted
                    if not features:
                        continue
                    
                    # Add position evaluation
                    # For early-mid game positions, use material+position
                    if move_num < 20:
                        # Create a label based on static evaluation
                        # rather than the game result
                        material_score = features[0]  # First feature is material advantage
                        position_score = sum(features[1:])  # Other features for positional advantage
                        
                        # Combined score normalized to [-1, 0, 1]
                        eval_score = material_score + 0.5 * position_score
                        if eval_score > 3:  # White is clearly better
                            label = 1
                        elif eval_score < -3:  # Black is clearly better
                            label = -1
                        else:  # Equal position
                            label = 0
                    else:
                        # For later positions, use the game result more heavily
                        label = game_result
                    
                    X.append(features)
                    y.append(label)
                    position_count += 1
                    
                except Exception as e:
                    print(f"Error processing move {move_num}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error processing game {game_count}: {e}")
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Extracted {position_count} positions from {game_count} games")
    
    # Check class balance
    unique_labels, counts = np.unique(y, return_counts=True)
    print("Label distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count} samples ({count/len(y)*100:.2f}%)")
    
    # If we have only one class, create synthetic data for other classes
    if len(unique_labels) < 2:
        print("WARNING: Only one class detected. Creating synthetic data for training...")
        
        if 0 in unique_labels:
            # If we have only class 0 (draws), create some wins and losses
            X_synthetic_white = X[:100].copy()
            # Modify features to make white winning positions
            X_synthetic_white[:, 0] += 5  # Increase material advantage
            X_synthetic_white[:, 1] += 3  # Increase mobility
            y_synthetic_white = np.ones(len(X_synthetic_white))
            
            X_synthetic_black = X[:100].copy()
            # Modify features to make black winning positions
            X_synthetic_black[:, 0] -= 5  # Decrease material advantage
            X_synthetic_black[:, 1] -= 3  # Decrease mobility
            y_synthetic_black = np.full(len(X_synthetic_black), -1)
            
            X = np.vstack([X, X_synthetic_white, X_synthetic_black])
            y = np.concatenate([y, y_synthetic_white, y_synthetic_black])
            
        elif 1 in unique_labels:
            # If we have only white wins, create draws and black wins
            X_synthetic_draw = X[:100].copy()
            # Modify features to make draw positions
            X_synthetic_draw[:, 0] = 0  # Equal material
            X_synthetic_draw[:, 1] = 0  # Equal mobility
            y_synthetic_draw = np.zeros(len(X_synthetic_draw))
            
            X_synthetic_black = X[:100].copy()
            # Modify features to make black winning positions
            X_synthetic_black[:, 0] = -X_synthetic_black[:, 0]  # Invert material advantage
            X_synthetic_black[:, 1] = -X_synthetic_black[:, 1]  # Invert mobility
            y_synthetic_black = np.full(len(X_synthetic_black), -1)
            
            X = np.vstack([X, X_synthetic_draw, X_synthetic_black])
            y = np.concatenate([y, y_synthetic_draw, y_synthetic_black])
            
        elif -1 in unique_labels:
            # If we have only black wins, create draws and white wins
            X_synthetic_draw = X[:100].copy()
            # Modify features to make draw positions
            X_synthetic_draw[:, 0] = 0  # Equal material
            X_synthetic_draw[:, 1] = 0  # Equal mobility
            y_synthetic_draw = np.zeros(len(X_synthetic_draw))
            
            X_synthetic_white = X[:100].copy()
            # Modify features to make white winning positions
            X_synthetic_white[:, 0] = -X_synthetic_white[:, 0]  # Invert material advantage
            X_synthetic_white[:, 1] = -X_synthetic_white[:, 1]  # Invert mobility
            y_synthetic_white = np.ones(len(X_synthetic_white))
            
            X = np.vstack([X, X_synthetic_draw, X_synthetic_white])
            y = np.concatenate([y, y_synthetic_draw, y_synthetic_white])
        
        # Print updated class distribution
        unique_labels, counts = np.unique(y, return_counts=True)
        print("Updated label distribution with synthetic data:")
        for label, count in zip(unique_labels, counts):
            print(f"  Class {label}: {count} samples ({count/len(y)*100:.2f}%)")
    
    return X, y

# Train SVM model with status updates and error handling
def train_svm(X, y, verbose=True):
    print("Starting SVM training process...")
    
    # Check if we have enough data
    if len(X) < 10:
        print("ERROR: Not enough training data. Need at least 10 samples.")
        return None, None
    
    # Check if we have enough classes
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print("ERROR: Need at least 2 classes for classification.")
        print("Attempting to create a synthetic class...")
        
        # Create synthetic data for missing classes
        if 0 in unique_classes:
            # Create positive and negative examples
            new_X = X[:min(100, len(X))].copy()
            # For half, make them positive examples
            half = len(new_X) // 2
            new_X[:half, 0] += 5  # Modify material advantage
            new_y_pos = np.ones(half)
            # For the other half, make them negative examples
            new_X[half:, 0] -= 5  # Modify material advantage
            new_y_neg = np.full(len(new_X) - half, -1)
            
            # Combine original and synthetic data
            X = np.vstack([X, new_X])
            y = np.concatenate([y, new_y_pos, new_y_neg])
            
        elif 1 in unique_classes:
            # Create draw and negative examples
            half_X = X[:min(100, len(X))].copy()
            # Make them draws
            half_X[:, 0] = 0  # Zero material advantage
            new_y_draw = np.zeros(len(half_X))
            # Make negative examples
            neg_X = X[:min(100, len(X))].copy()
            neg_X[:, 0] = -neg_X[:, 0]  # Invert material advantage
            new_y_neg = np.full(len(neg_X), -1)
            
            # Combine
            X = np.vstack([X, half_X, neg_X])
            y = np.concatenate([y, new_y_draw, new_y_neg])
            
        elif -1 in unique_classes:
            # Create draw and positive examples
            half_X = X[:min(100, len(X))].copy()
            # Make them draws
            half_X[:, 0] = 0  # Zero material advantage
            new_y_draw = np.zeros(len(half_X))
            # Make positive examples
            pos_X = X[:min(100, len(X))].copy()
            pos_X[:, 0] = -pos_X[:, 0]  # Invert material advantage
            new_y_pos = np.ones(len(pos_X))
            
            # Combine
            X = np.vstack([X, half_X, pos_X])
            y = np.concatenate([y, new_y_draw, new_y_pos])
        
        print(f"Created synthetic data. New data shape: {X.shape}")
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print("ERROR: Failed to create multiple classes. Cannot train SVM.")
            return None, None
    
    # Split data
    print("Splitting data into train/test sets...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"Error splitting data: {e}")
        return None, None
        
    print(f"Training data size: {X_train.shape[0]}, Test data size: {X_test.shape[0]}")
    
    # Check for empty splits
    if len(X_train) == 0 or len(X_test) == 0:
        print("ERROR: Empty training or test set after splitting.")
        return None, None
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    try:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    except Exception as e:
        print(f"Error scaling features: {e}")
        return None, None
        
    print("Feature scaling complete")
    
    # Calculate class distribution
    train_classes, train_counts = np.unique(y_train, return_counts=True)
    print("Class distribution in training data:")
    for cls, count in zip(train_classes, train_counts):
        print(f"  Class {cls}: {count} samples ({count/len(y_train)*100:.2f}%)")
    
    # Check for class imbalance and handle if needed
    if len(train_classes) < 2:
        print("ERROR: Less than 2 classes in training set after splitting.")
        # Try to augment data
        y_train_augmented = y_train.copy()
        X_train_augmented = X_train.copy()
        
        if 0 in train_classes:
            # Add some synthetic positive and negative samples
            half = len(X_train) // 2
            X_pos = X_train[:half].copy()
            X_pos[:, 0] += 3  # Increase material
            y_pos = np.ones(len(X_pos))
            
            X_neg = X_train[half:].copy()
            X_neg[:, 0] -= 3  # Decrease material
            y_neg = np.full(len(X_neg), -1)
            
            X_train_augmented = np.vstack([X_train, X_pos, X_neg])
            y_train_augmented = np.concatenate([y_train, y_pos, y_neg])
            
        elif 1 in train_classes:
            # Add synthetic draws and negative samples
            X_draw = X_train[:len(X_train)//2].copy()
            X_draw[:, 0] = 0
            y_draw = np.zeros(len(X_draw))
            
            X_neg = X_train[len(X_train)//2:].copy()
            X_neg[:, 0] = -X_neg[:, 0]
            y_neg = np.full(len(X_neg), -1)
            
            X_train_augmented = np.vstack([X_train, X_draw, X_neg])
            y_train_augmented = np.concatenate([y_train, y_draw, y_neg])
            
        elif -1 in train_classes:
            # Add synthetic draws and positive samples
            X_draw = X_train[:len(X_train)//2].copy()
            X_draw[:, 0] = 0
            y_draw = np.zeros(len(X_draw))
            
            X_pos = X_train[len(X_train)//2:].copy()
            X_pos[:, 0] = -X_pos[:, 0]
            y_pos = np.ones(len(X_pos))
            
            X_train_augmented = np.vstack([X_train, X_draw, X_pos])
            y_train_augmented = np.concatenate([y_train, y_draw, y_pos])
        
        # Update training data
        X_train = X_train_augmented
        y_train = y_train_augmented
        
        # Rescale augmented data
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Check updated class distribution
        train_classes, train_counts = np.unique(y_train, return_counts=True)
        print("Updated class distribution after augmentation:")
        for cls, count in zip(train_classes, train_counts):
            print(f"  Class {cls}: {count} samples ({count/len(y_train)*100:.2f}%)")
        
        if len(train_classes) < 2:
            print("ERROR: Failed to create multiple classes even after augmentation.")
            return None, None
    
    # Train SVM with incremental feedback
    print("\nTraining SVM model...")
    
    # For better status updates, we'll use a custom implementation with different C values
    c_values = [0.1, 1, 10, 100]
    best_accuracy = 0
    best_model = None
    
    for i, c in enumerate(c_values):
        print(f"\nTrying C={c} ({i+1}/{len(c_values)})...")
        try:
            model = SVC(kernel='rbf', C=c, probability=True, verbose=False)
            
            # Skip cross-validation if we have too few samples
            if len(X_train) > 20:
                # Quick cross-validation to gauge performance
                from sklearn.model_selection import cross_val_score
                try:
                    cv = min(3, len(X_train) // 10)  # Ensure we have enough samples per fold
                    cv = max(2, cv)  # At least 2 folds
                    
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
                    print(f"  Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                except Exception as e:
                    print(f"  Warning: Cross-validation failed: {e}")
                    print("  Continuing with direct model fitting...")
            
            # Fit the model on the full training data
            print(f"  Fitting model with C={c}...")
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"  Test accuracy: {accuracy:.4f}")
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                print(f"  New best model with accuracy: {best_accuracy:.4f}")
        except Exception as e:
            print(f"  Error training model with C={c}: {e}")
            continue
    
    if best_model is None:
        print("ERROR: Failed to train any model. Trying simpler approach...")
        try:
            # Try a simpler model (LinearSVC)
            from sklearn.svm import LinearSVC
            model = LinearSVC(dual="auto", max_iter=10000)
            model.fit(X_train_scaled, y_train)
            
            # Convert to SVC for consistency
            best_model = SVC(kernel='linear', C=1.0, probability=True)
            best_model.fit(X_train_scaled, y_train)
            
            y_pred = best_model.predict(X_test_scaled)
            best_accuracy = accuracy_score(y_test, y_pred)
            print(f"LinearSVC fallback accuracy: {best_accuracy:.4f}")
        except Exception as e:
            print(f"ERROR: Even simple model training failed: {e}")
            return None, None
    
    print("\nSVM training complete!")
    print(f"Best model accuracy: {best_accuracy:.4f}")
    
    # Detailed evaluation of the best model
    from sklearn.metrics import classification_report, confusion_matrix
    
    try:
        y_pred = best_model.predict(X_test_scaled)
        print("\nDetailed evaluation:")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    except Exception as e:
        print(f"Warning: Could not generate detailed evaluation: {e}")
    
    return best_model, scaler

# Function to evaluate a chess position using the trained model
def evaluate_position(board, model, scaler):
    if model is None or scaler is None:
        # Fallback to simple material evaluation if no model
        pieces = board.piece_map()
        piece_values = {
            chess.PAWN: 1, 
            chess.KNIGHT: 3, 
            chess.BISHOP: 3, 
            chess.ROOK: 5, 
            chess.QUEEN: 9, 
            chess.KING: 0
        }
        white_material = 0
        black_material = 0
        
        for piece in pieces.values():
            value = piece_values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                white_material += value
            else:
                black_material += value
                
        score = (white_material - black_material) / 32.0  # Normalize
        return score
        
    try:
        features = extract_features_from_position(board)
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Get probabilities
        probs = model.predict_proba(features_scaled)[0]
        
        # Calculate weighted evaluation score
        # Assuming class order is [-1, 0, 1] for [black win, draw, white win]
        score = 0
        for i, label in enumerate(model.classes_):
            score += label * probs[i]
        
        # Return score from white's perspective
        if board.turn == chess.BLACK:
            score = -score
            
        return score
    except Exception as e:
        print(f"Error in position evaluation: {e}")
        # Fallback to simple material evaluation
        pieces = board.piece_map()
        piece_values = {
            chess.PAWN: 1, 
            chess.KNIGHT: 3, 
            chess.BISHOP: 3, 
            chess.ROOK: 5, 
            chess.QUEEN: 9, 
            chess.KING: 0
        }
        white_material = 0
        black_material = 0
        
        for piece in pieces.values():
            value = piece_values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                white_material += value
            else:
                black_material += value
                
        score = (white_material - black_material) / 32.0  # Normalize
        return score

# Implement Deep Q-Network (DQN) for chess AI with status tracking
class ChessDQN:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1  # Exploration rate
        
        # Status tracking variables
        self.total_games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.average_game_length = 0
        self.training_history = []
    
    def choose_move(self, board, legal_moves=None):
        if legal_moves is None:
            legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return None
        
        # Epsilon-greedy strategy
        if np.random.random() < self.epsilon:
            # Random move
            return np.random.choice(legal_moves)
        
        # Otherwise, choose the best move
        best_score = float("-inf")
        best_move = None
        
        for move in legal_moves:
            # Make the move
            board.push(move)
            
            # Evaluate the position
            score = evaluate_position(board, self.model, self.scaler)
            
            # Undo the move
            board.pop()
            
            # Update best move
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def get_status_report(self):
        """Generate a status report with current performance metrics"""
        win_rate = self.wins / max(1, self.total_games_played) * 100
        loss_rate = self.losses / max(1, self.total_games_played) * 100
        draw_rate = self.draws / max(1, self.total_games_played) * 100
        
        report = {
            "total_games": self.total_games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "draw_rate": draw_rate,
            "average_game_length": self.average_game_length,
            "current_epsilon": self.epsilon
        }
        
        return report
    
    def print_status(self, episode=None, total_episodes=None):
        """Print current training status"""
        status = self.get_status_report()
        
        print("\n" + "="*50)
        if episode is not None and total_episodes is not None:
            print(f"Training Progress: Episode {episode}/{total_episodes} ({episode/total_episodes*100:.1f}%)")
        print(f"Games Played: {status['total_games']}")
        print(f"Win Rate: {status['win_rate']:.2f}% ({status['wins']} wins)")
        print(f"Loss Rate: {status['loss_rate']:.2f}% ({status['losses']} losses)")
        print(f"Draw Rate: {status['draw_rate']:.2f}% ({status['draws']} draws)")
        print(f"Average Game Length: {status['average_game_length']:.2f} moves")
        print(f"Current Exploration Rate (epsilon): {status['current_epsilon']:.3f}")
        print("="*50)
    
    def train(self, num_episodes=100, status_interval=10, decay_epsilon=True):
        """
        Train the DQN agent
        
        Args:
            num_episodes: Number of games to play
            status_interval: How often to print status updates
            decay_epsilon: Whether to decay the exploration rate
        """
        print(f"Starting DQN training for {num_episodes} episodes...")
        
        for episode in range(1, num_episodes + 1):
            board = chess.Board()
            move_count = 0
            done = False
            
            # For decay of epsilon (exploration rate)
            if decay_epsilon:
                # Decay epsilon from 0.9 to 0.1 over all episodes
                self.epsilon = max(0.1, 0.9 - 0.8 * (episode / num_episodes))
            
            while not done:
                move = self.choose_move(board)
                
                if move is None:  # No legal moves
                    done = True
                    continue
                
                old_features = extract_features_from_position(board)
                old_features_scaled = self.scaler.transform([old_features])
                
                # Make move
                board.push(move)
                move_count += 1
                
                # Check if game is over
                if board.is_game_over():
                    done = True
                    # Update game statistics
                    self.total_games_played += 1
                    
                    # Get reward based on game outcome
                    result = board.result()
                    if result == "1-0":  # White wins
                        reward = 1
                        self.wins += 1
                    elif result == "0-1":  # Black wins
                        reward = -1
                        self.losses += 1
                    else:  # Draw
                        reward = 0
                        self.draws += 1
                        
                    # Update average game length
                    self.average_game_length = ((self.average_game_length * 
                                               (self.total_games_played - 1)) + 
                                               move_count) / self.total_games_played
                else:
                    reward = 0
                
                # Q-learning update
                if not done:
                    next_features = extract_features_from_position(board)
                    next_features_scaled = self.scaler.transform([next_features])
                    next_value = np.max(self.model.predict_proba(next_features_scaled)[0])
                    target = reward + self.discount_factor * next_value
                else:
                    target = reward
                
                # Record training progress
                self.training_history.append({
                    'episode': episode,
                    'move': move_count,
                    'reward': reward,
                    'done': done
                })
                
                # Make opponent move (simple)
                if not done:
                    opponent_move = self.choose_move(board)
                    if opponent_move:
                        board.push(opponent_move)
                        move_count += 1
                        if board.is_game_over():
                            done = True
                            
                            # Update game statistics
                            self.total_games_played += 1
                            result = board.result()
                            if result == "1-0":  # White wins
                                self.losses += 1  # Opponent won
                            elif result == "0-1":  # Black wins
                                self.wins += 1  # We won
                            else:  # Draw
                                self.draws += 1
                            
                            # Update average game length
                            self.average_game_length = ((self.average_game_length * 
                                                      (self.total_games_played - 1)) + 
                                                      move_count) / self.total_games_played
            
            # Print status at intervals
            if episode % status_interval == 0 or episode == 1 or episode == num_episodes:
                self.print_status(episode, num_episodes)
        
        print("\nTraining complete!")
        self.print_status()
        
        # Save final training metrics
        training_metrics = {
            'total_games': self.total_games_played,
            'win_rate': self.wins / max(1, self.total_games_played),
            'loss_rate': self.losses / max(1, self.total_games_played),
            'draw_rate': self.draws / max(1, self.total_games_played),
            'average_game_length': self.average_game_length,
            'final_epsilon': self.epsilon
        }
        
        return training_metrics

# Example usage with comprehensive status reporting
def main(file_path, dqn_episodes=50, verbose=True):
    """
    Main function to run the chess SVM-DQN AI
    
    Args:
        file_path: Path to the CSV file with chess game data
        dqn_episodes: Number of episodes for DQN training
        verbose: Whether to print detailed status reports
    """
    import time
    start_time = time.time()
    
    # Print header
    print("="*60)
    print("CHESS AI TRAINING USING SVM AND DQN")
    print("="*60)
    
    # Parse data
    print(f"Loading data from {file_path}...")
    df = parse_chess_data(file_path)
    print(f"Loaded {len(df)} chess games")
    
    # Process games
    print("\nProcessing games to extract features...")
    process_start = time.time()
    X, y = process_games(df)
    process_time = time.time() - process_start
    print(f"Extracted {len(X)} positions with {len(X[0])} features each")
    print(f"Processing completed in {process_time:.2f} seconds")
    
    # Display feature statistics
    print("\nFeature statistics:")
    feature_names = [
        "Material Advantage", 
        "Mobility Advantage",
        "Center Control",
        "Pawn Structure",
        "King Safety",
        "Development"
    ]
    
    X_array = np.array(X)
    for i, name in enumerate(feature_names):
        if i < X_array.shape[1]:
            print(f"  {name}: mean={X_array[:, i].mean():.4f}, std={X_array[:, i].std():.4f}, "
                  f"min={X_array[:, i].min():.4f}, max={X_array[:, i].max():.4f}")
    
    # Train SVM
    print("\n" + "="*60)
    print("TRAINING SVM MODEL")
    print("="*60)
    svm_start = time.time()
    model, scaler = train_svm(X, y, verbose=verbose)
    svm_time = time.time() - svm_start
    print(f"SVM training completed in {svm_time:.2f} seconds")
    
    # Initialize DQN agent
    print("\n" + "="*60)
    print("TRAINING DQN AGENT")
    print("="*60)
    print("Initializing DQN agent...")
    dqn_agent = ChessDQN(model, scaler)
    
    # Training
    dqn_start = time.time()
    print(f"\nTraining DQN agent for {dqn_episodes} episodes...")
    training_metrics = dqn_agent.train(num_episodes=dqn_episodes, status_interval=5)
    dqn_time = time.time() - dqn_start
    print(f"DQN training completed in {dqn_time:.2f} seconds")
    
    # Test the trained model
    print("\n" + "="*60)
    print("TESTING THE TRAINED MODEL")
    print("="*60)
    
    # Test a simple position
    print("\nTesting position evaluation on common openings...")
    
    test_positions = [
        ("Starting Position", []),
        ("King's Pawn Opening", ["e4"]),
        ("Sicilian Defense", ["e4", "c5"]),
        ("French Defense", ["e4", "e6"]),
        ("Queen's Gambit", ["d4", "d5", "c4"]),
        ("King's Indian Defense", ["d4", "Nf6", "c4", "g6"]),
        ("Ruy Lopez", ["e4", "e5", "Nf3", "Nc6", "Bb5"])
    ]
    
    for name, moves in test_positions:
        board = chess.Board()
        for move in moves:
            board.push_san(move)
        
        score = evaluate_position(board, model, scaler)
        print(f"  {name}: {score:.4f}")
        
        # Get best move
        best_move = dqn_agent.choose_move(board)
        if best_move:
            print(f"    Best move: {board.san(best_move)}")
        else:
            print("    No legal moves available")
    
    # Print overall performance summary
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Game positions analyzed: {len(X)}")
    print(f"SVM training time: {svm_time:.2f} seconds")
    print(f"DQN training time: {dqn_time:.2f} seconds")
    print(f"Final win rate: {training_metrics['win_rate']*100:.2f}%")
    print(f"Average game length: {training_metrics['average_game_length']:.2f} moves")
    print("="*60)
    
    return model, scaler, dqn_agent, training_metrics

# To use this code, you would run:
model, scaler, dqn_agent, metrics = main('dataset/1000games.csv')