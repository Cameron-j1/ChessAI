import pandas as pd
import numpy as np
import chess.pgn
import io
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def extract_game_features(game):
    """Extract relevant features from a chess game."""
    # Basic game info
    features = {}
    
    # Extract result
    result = game.headers.get("Result", "*")
    if result == "1-0":
        features["result"] = "white"
    elif result == "0-1":
        features["result"] = "black"
    elif result == "1/2-1/2":
        features["result"] = "draw"
    else:
        features["result"] = None
    
    # Extract player ratings
    features["white_rating"] = int(game.headers.get("WhiteElo", 0)) or None
    features["black_rating"] = int(game.headers.get("BlackElo", 0)) or None
    features["rating_diff"] = features["white_rating"] - features["black_rating"] if features["white_rating"] and features["black_rating"] else None
    
    # Extract time control
    time_control = game.headers.get("TimeControl", "")
    if "+" in time_control:
        base_time, increment = time_control.split("+")
        features["base_time"] = int(base_time) if base_time.isdigit() else None
        features["increment"] = int(increment) if increment.isdigit() else None
    else:
        features["base_time"] = int(time_control) if time_control.isdigit() else None
        features["increment"] = 0
    
    # Extract ECO code (chess opening)
    features["eco_code"] = game.headers.get("ECO", "")
    
    # Count moves and captures by piece type
    piece_moves = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}  # 1=pawn, 2=knight, 3=bishop, 4=rook, 5=queen, 6=king
    piece_captures = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    
    # Navigate through the moves
    board = game.board()
    num_moves = 0
    node = game
    
    while node.variations:
        next_node = node.variation(0)
        move = next_node.move
        if move:
            num_moves += 1
            piece_type = board.piece_type_at(move.from_square)
            if piece_type:
                piece_moves[piece_type] += 1
                if board.piece_at(move.to_square):
                    piece_captures[piece_type] += 1
            board.push(move)
        node = next_node
    
    features["num_moves"] = num_moves
    
    # Add piece move and capture counts to features
    for piece_type in range(1, 7):
        features[f"piece_{piece_type}_moves"] = piece_moves[piece_type]
        features[f"piece_{piece_type}_captures"] = piece_captures[piece_type]
    
    return features

def parse_pgn_file(file_content):
    """Parse a PGN file and extract games."""
    pgn_data = io.StringIO(file_content)
    games = []
    game_count = 0
    
    while True:
        game = chess.pgn.read_game(pgn_data)
        if game is None:
            break
        games.append(game)
        game_count += 1
    
    print(f"Found {game_count} games in file")
    return games

def extract_features_from_games(games):
    """Extract features from all games."""
    game_features = []
    valid_games = 0
    
    for game in games:
        features = extract_game_features(game)
        if features["result"] is not None:
            game_features.append(features)
            valid_games += 1
    
    print(f"Successfully parsed {valid_games} games")
    return game_features

def prepare_data_for_svm(game_features):
    """Prepare data for SVM model."""
    print("Preparing data for SVM...")
    df = pd.DataFrame(game_features)
    
    # Remove rows with missing values (simplistic approach)
    df = df.dropna()
    
    # Separate features and target
    X = df.drop('result', axis=1)
    y = df['result']
    
    # Identify categorical features (like ECO code)
    categorical_features = ['eco_code']
    numeric_features = [col for col in X.columns if col not in categorical_features]
    
    print(f"Features shape: {X.shape}")
    print(f"Feature columns: {list(X.columns)[:15]}...")
    
    print("\nClass distribution:")
    print(y.value_counts())
    
    return X, y, categorical_features, numeric_features

def train_svm_model(X, y, categorical_features, numeric_features):
    """Train an SVM model with proper preprocessing."""
    print("\nClass distribution:")
    print(y.value_counts())
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create preprocessing pipelines for both numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create a preprocessing and classification pipeline
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(random_state=42))
    ])
    
    # Define a parameter grid for GridSearchCV
    param_grid = {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__gamma': ['scale', 'auto'],
        'classifier__kernel': ['rbf', 'linear', 'poly', 'sigmoid']
    }
    
    # Perform grid search with cross-validation
    print("Training SVM model with GridSearchCV...")
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    
    try:
        grid_search.fit(X_train, y_train)
        print("Best parameters:", grid_search.best_params_)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Evaluate the model
        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        print("Confusion matrix saved as 'confusion_matrix.png'")
        
        return best_model, (X_train, X_test, y_train, y_test), grid_search
        
    except Exception as e:
        print(f"Error during analysis: \n{str(e)}")
        raise

def explain_model_predictions(model, X_test, y_test):
    """Explain model predictions for a few examples."""
    # Get feature names after preprocessing
    feature_names = []
    for name, transformer, features in model.named_steps['preprocessor'].transformers_:
        if hasattr(transformer, 'get_feature_names_out'):
            names = transformer.get_feature_names_out(features)
        else:
            names = features
        feature_names.extend(names)
    
    # Get model coefficients if it's a linear model
    if hasattr(model.named_steps['classifier'], 'coef_'):
        coefficients = model.named_steps['classifier'].coef_[0]
        feature_importance = dict(zip(feature_names, coefficients))
        
        # Sort by absolute value
        sorted_importance = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print("\nTop 10 most important features:")
        for feature, importance in sorted_importance[:10]:
            print(f"{feature}: {importance:.4f}")
    
    # Predict a few examples and explain
    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
    
    print("\nExample predictions:")
    for idx in sample_indices:
        sample = X_test.iloc[idx:idx+1]
        true_label = y_test.iloc[idx]
        pred_label = model.predict(sample)[0]
        
        print(f"\nExample {idx}:")
        print(f"True outcome: {true_label}")
        print(f"Predicted outcome: {pred_label}")
        print("Key features:")
        for col in ['white_rating', 'black_rating', 'rating_diff', 'eco_code', 'num_moves']:
            if col in sample:
                print(f"  {col}: {sample[col].values[0]}")

def load_pgn_from_file(file_path):
    """Load PGN data from a file."""
    with open(file_path, 'r') as f:
        return f.read()

def main():
    print("Starting Chess PGN SVM Analysis...")
    
    try:
        # Load PGN data (replace with your file path)
        pgn_content = load_pgn_from_file("dataset/standardover2000-2021.pgn")
        
        # Parse PGN data
        games = parse_pgn_file(pgn_content)
        
        # Extract features
        game_features = extract_features_from_games(games)
        
        # Prepare data
        X, y, categorical_features, numeric_features = prepare_data_for_svm(game_features)
        
        # Train model
        print("Training SVM model...")
        model, (X_train, X_test, y_train, y_test), grid_search = train_svm_model(
            X, y, categorical_features, numeric_features
        )
        
        # Explain predictions
        explain_model_predictions(model, X_test, y_test)
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()