import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
import chess.pgn
import io

# Function to read the CSV file
def load_chess_data(filepath):
    try:
        # Try to load with first row as header
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Successfully loaded data with {len(df)} rows")
        return df
    except Exception as e:
        try:
            # If that fails, try loading without headers
            df = pd.read_csv(filepath, header=None, low_memory=False)
            print(f"Successfully loaded data with {len(df)} rows (no headers)")
            
            # Check if first row looks like headers
            first_row = df.iloc[0]
            if isinstance(first_row[0], str) and first_row[0] in ['game_id', 'id', 'd']:
                print("First row appears to be headers, removing it...")
                df = df.iloc[1:].reset_index(drop=True)
            
            return df
        except Exception as e2:
            print(f"Error loading data: {str(e2)}")
            return None

# Function to extract features from PGN
def extract_features_from_pgn(pgn_text):
    """Extract useful features from PGN text."""
    features = {}
    
    # Ensure pgn_text is a string
    if not isinstance(pgn_text, str):
        print(f"Warning: pgn_text is not a string but {type(pgn_text)}. Converting to string.")
        pgn_text = str(pgn_text)
    
    # Clean the PGN text - replace HTML entities and remove non-standard characters
    pgn_text = pgn_text.replace('&quot;', '"').replace('&amp;', '&')
    
    # Number of moves in the game
    moves = re.findall(r'\d+\.', pgn_text)
    features['num_moves'] = len(moves)
    
    # Check if certain openings are played (based on first few moves)
    features['e4_opening'] = 1 if re.search(r'\b[^a-zA-Z]e4\b', pgn_text) is not None else 0
    features['d4_opening'] = 1 if re.search(r'\b[^a-zA-Z]d4\b', pgn_text) is not None else 0
    features['c4_opening'] = 1 if re.search(r'\b[^a-zA-Z]c4\b', pgn_text) is not None else 0
    features['nf3_opening'] = 1 if re.search(r'\b[^a-zA-Z]Nf3\b', pgn_text) is not None else 0
    
    # Capture frequency
    features['captures'] = pgn_text.count('x')
    
    # Check frequency
    features['checks'] = pgn_text.count('+')
    
    # Castling
    features['castling_kingside'] = 1 if 'O-O' in pgn_text else 0
    features['castling_queenside'] = 1 if 'O-O-O' in pgn_text else 0
    
    # Pawn promotion frequency
    features['promotions'] = len(re.findall(r'=[QRNB]', pgn_text))
    
    # Advanced piece developments
    features['knight_development'] = len(re.findall(r'N[a-h]', pgn_text))
    features['bishop_development'] = len(re.findall(r'B[a-h]', pgn_text))
    features['queen_moves'] = len(re.findall(r'Q[a-h\d]', pgn_text))
    features['rook_moves'] = len(re.findall(r'R[a-h\d]', pgn_text))
    
    # Game length (total character length of PGN might correlate with game complexity)
    features['pgn_length'] = len(pgn_text)
    
    # Additional features
    
    # Count number of captures for each piece type
    features['pawn_captures'] = len(re.findall(r'[a-h]x', pgn_text))
    features['bishop_captures'] = len(re.findall(r'Bx', pgn_text))
    features['knight_captures'] = len(re.findall(r'Nx', pgn_text))
    features['rook_captures'] = len(re.findall(r'Rx', pgn_text))
    features['queen_captures'] = len(re.findall(r'Qx', pgn_text))
    
    # Center control (moves to e4, d4, e5, d5)
    features['center_control'] = sum([
        pgn_text.count(' e4 '), pgn_text.count(' d4 '),
        pgn_text.count(' e5 '), pgn_text.count(' d5 ')
    ])
    
    # Look for common tactical patterns
    features['pin_skewer_potential'] = pgn_text.count('Bg5') + pgn_text.count('Bf4') + pgn_text.count('Bb5')
    features['fork_potential'] = pgn_text.count('Nc3') + pgn_text.count('Nf3') + pgn_text.count('Nd4') + pgn_text.count('Nb5')
    
    return features

# Function to process the chess data and prepare for SVM
def process_chess_data(df):
    """Process chess data and extract features for SVM."""
    # Print sample data to understand structure
    print("Sample row from dataframe:")
    print(df.iloc[0])
    
    # Try to infer correct column names based on the data
    expected_columns = ['game_id', 'tournament', 'timestamp_start', 'timestamp_end', 
                       'moves_count', 'termination', 'winner', 'time_control', 
                       'white_player', 'white_rating', 'black_player', 'black_rating', 
                       'pgn', 'eco_code', 'opening_name', 'opening_ply']
    
    # If we have fewer columns than expected, adjust
    if len(df.columns) < len(expected_columns):
        expected_columns = expected_columns[:len(df.columns)]
    
    # If the dataframe already has named columns, don't rename them
    if not all(col.isdigit() for col in [str(col) for col in df.columns]) and not all(isinstance(col, int) for col in df.columns):
        print("DataFrame already has named columns, keeping them")
    else:
        # Otherwise, rename columns
        df.columns = expected_columns[:len(df.columns)]
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Identify PGN column
    pgn_col = None
    for col in df.columns:
        # Check a sample of values in the column to see if they look like PGN
        sample = df[col].astype(str).sample(min(5, len(df))).tolist()
        pgn_patterns = ['e4', 'd4', 'Nf3', 'O-O', 'Qxd', 'Bxe', 'Rad']
        if any(any(pattern in str(s) for pattern in pgn_patterns) for s in sample):
            pgn_col = col
            print(f"Identified likely PGN column: {pgn_col}")
            break
    
    if pgn_col is None:
        # Fall back to using the expected PGN column name
        pgn_col = 'pgn' if 'pgn' in df.columns else 'moves'
        print(f"Using column '{pgn_col}' for PGN data")
    
    # Extract features from PGN
    print("Extracting features from PGN...")
    features_list = []
    for pgn in df[pgn_col]:
        features = extract_features_from_pgn(pgn)
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    print(f"Extracted features shape: {features_df.shape}")
    
    # Try to find rating columns
    white_rating_col = None
    black_rating_col = None
    
    # Check for standard rating column names
    if 'white_rating' in df.columns:
        white_rating_col = 'white_rating'
    if 'black_rating' in df.columns:
        black_rating_col = 'black_rating'
    
    # If not found, try to infer them
    if white_rating_col is None or black_rating_col is None:
        for col in df.columns:
            col_str = str(col).lower()
            if 'white' in col_str and ('rating' in col_str or 'elo' in col_str):
                white_rating_col = col
            if 'black' in col_str and ('rating' in col_str or 'elo' in col_str):
                black_rating_col = col
    
    # If found, calculate rating difference
    if white_rating_col is not None and black_rating_col is not None:
        try:
            # Convert ratings to numeric and add player rating difference as a feature
            df['white_rating_num'] = pd.to_numeric(df[white_rating_col], errors='coerce')
            df['black_rating_num'] = pd.to_numeric(df[black_rating_col], errors='coerce')
            df['rating_diff'] = df['white_rating_num'] - df['black_rating_num']
            
            # Combine with features extracted from PGN
            X = pd.concat([features_df, df[['rating_diff']]], axis=1)
        except Exception as e:
            print(f"Error calculating rating difference: {str(e)}")
            X = features_df
    else:
        print("Could not find white or black rating columns, skipping rating difference calculation")
        X = features_df
    
    # Define target variable: winner (white/black/draw)
    # Find winner column
    winner_col = None
    if 'winner' in df.columns:
        winner_col = 'winner'
    else:
        for col in df.columns:
            if str(col).lower() == 'winner' or 'result' in str(col).lower():
                winner_col = col
                break
    
    if winner_col:
        y = df[winner_col]
        print(f"Using '{winner_col}' as target variable")
    else:
        print("Could not find winner column, using first column as target")
        y = df.iloc[:, 0]
    
    # Handle missing values if any
    X = X.fillna(0)
    
    print(f"Final features shape: {X.shape}")
    print(f"Feature columns: {X.columns.tolist()}")
    
    return X, y

# Function to train SVM model
def train_svm_model(X, y):
    """Train SVM model with grid search for optimal hyperparameters."""
    # Print class distribution
    print("\nClass distribution:")
    class_dist = y.value_counts()
    print(class_dist)
    
    # Check for class imbalance
    min_class_size = class_dist.min()
    if min_class_size < 10:
        print(f"Warning: Minimum class size is only {min_class_size}. Consider handling class imbalance.")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Create a pipeline with scaling (important for SVM)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, class_weight='balanced'))
    ])
    
    # Define hyperparameters to search
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.1, 0.01],
        'svm__kernel': ['rbf', 'linear']
    }
    
    # Grid search with cross-validation
    print("Training SVM model with GridSearchCV...")
    # Use fewer folds if a class has very few members
    cv = min(5, min_class_size)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    return best_model, (X_train, X_test, y_train, y_test), grid_search

# Function to analyze feature importance (for linear kernel)
def analyze_feature_importance(model, feature_names):
    """Analyze feature importance for linear SVM."""
    if hasattr(model[-1], 'coef_'):
        # For linear kernel
        importance = np.abs(model[-1].coef_).mean(axis=0)
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
        plt.title('Top 15 Feature Importance')
        plt.tight_layout()
        
        return feature_importance
    else:
        print("Feature importance is only available for linear kernel.")
        return None

# Main function
def main(filepath):
    print("Starting Chess Games SVM Analysis...")
    
    # Load data
    df = load_chess_data(filepath)
    if df is None:
        print("Failed to load data. Please check the file path.")
        return
    
    try:
        # Process data
        X, y = process_chess_data(df)
        
        # Check for minimum samples per class
        class_counts = y.value_counts()
        print("\nClass distribution:")
        print(class_counts)
        
        if class_counts.min() < 2:
            print("Warning: Some classes have fewer than 2 samples, which is insufficient for training.")
            print("Filtering out rare classes...")
            # Keep only classes with at least 5 samples
            valid_classes = class_counts[class_counts >= 5].index
            if len(valid_classes) >= 2:
                mask = y.isin(valid_classes)
                X = X[mask]
                y = y[mask]
                print(f"Filtered data to {len(X)} samples in {len(valid_classes)} classes")
            else:
                print("Not enough samples in any class. Cannot proceed with training.")
                return
        
        # Train SVM model
        model, (X_train, X_test, y_train, y_test), grid_search = train_svm_model(X, y)
        
        # Analyze feature importance (for linear kernel)
        if model[-1].kernel == 'linear':
            feature_importance = analyze_feature_importance(model, X.columns)
            if feature_importance is not None:
                print("\nTop 10 important features:")
                print(feature_importance.head(10))
        
        # Print overall accuracy
        try:
            print(f"\nBest validation score: {grid_search.best_score_:.4f}")
            print(f"Test accuracy: {model.score(X_test, y_test):.4f}")
            
            # Additional evaluation metrics
            y_pred = model.predict(X_test)
            from sklearn.metrics import balanced_accuracy_score, roc_auc_score
            print(f"Balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
            
            # If there are more than 2 classes, use ROC AUC with OvR approach
            if len(np.unique(y)) > 2:
                try:
                    # Get probability predictions
                    y_score = model.predict_proba(X_test)
                    # Compute ROC AUC
                    roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr', average='weighted')
                    print(f"ROC AUC (weighted): {roc_auc:.4f}")
                except Exception as e:
                    print(f"Could not compute ROC AUC: {str(e)}")
            
        except Exception as e:
            print(f"Error calculating scores: {str(e)}")
        
        print("\nAnalysis complete!")
        return model
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Run the analysis
if __name__ == "__main__":
    # Replace with the actual file path from user input
    filepath = "dataset/1000games.csv"  # This should be updated with the actual filepath
    main(filepath)