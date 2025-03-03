import pandas as pd
import numpy as np
import os
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Load and preprocess the data
def load_and_preprocess_data(csv_path):
    """
    Load the football tracking data and preprocess it for MTGNN model
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Sort by time to ensure sequences are in order
    df = df.sort_values(by=['IdPeriod', 'Time'])
    
    return df

# 2. Feature extraction and node definition
def extract_features(df):
    """
    Extract relevant features for each node (players and ball)
    """
    # Define nodes: home players + away players + 1 ball
    
    # Extract features for each node
    node_features = []
    home_players = 0
    away_players = 0
    
    print("Extracting features for each node...")
    
    # Home players
    for i in range(1, 15):  # From the data structure, it seems there are up to 14 players
        if f'home_{i}_x' in df.columns and f'home_{i}_y' in df.columns:
            # For each player, extract position and distance to ball
            player_features = df[[
                f'home_{i}_x', 
                f'home_{i}_y',
                f'home_{i}_DistanceToBall'
            ]]
            node_features.append(player_features)
            home_players += 1
    
    print(f"Found {home_players} home players")
    
    # Away players
    for i in range(1, 15):
        if f'away_{i}_x' in df.columns and f'away_{i}_y' in df.columns:
            # For each player, extract position and distance to ball
            player_features = df[[
                f'away_{i}_x', 
                f'away_{i}_y',
                f'away_{i}_DistanceToBall'
            ]]
            node_features.append(player_features)
            away_players += 1
    
    print(f"Found {away_players} away players")
    
    # Check if we have the ball columns
    ball_columns = ['ball_x_Home', 'ball_y_Home', 'Ball_Speed', 'Ball_Acceleration', 'Ball_Direction_Degrees']
    missing_columns = [col for col in ball_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing ball columns: {missing_columns}")
        # Use only available ball columns
        ball_columns = [col for col in ball_columns if col in df.columns]
    
    # Ball - using both home and away perspectives
    if ball_columns:
        ball_features = df[ball_columns]
        node_features.append(ball_features)
        print("Added ball node with features:", ball_columns)
    else:
        print("Warning: No ball features found in the dataset")
    
    # Print information about the extracted features
    total_nodes = home_players + away_players + (1 if ball_columns else 0)
    print(f"Total nodes extracted: {total_nodes}")
    
    for i, features in enumerate(node_features):
        node_type = "Home player" if i < home_players else "Away player" if i < home_players + away_players else "Ball"
        print(f"Node {i} ({node_type}): {features.shape[1]} features, {features.shape[0]} time steps")
    
    # Normalize features to have similar scales
    print("Normalizing features...")
    normalized_features = []
    scaling_stats = []
    
    for features in node_features:
        # Handle missing data by filling with 0
        features = features.fillna(0)
        
        # Apply normalization
        scaler = StandardScaler()
        normalized = pd.DataFrame(
            scaler.fit_transform(features), 
            columns=features.columns
        )
        normalized_features.append(normalized)

        for col, mean, std in zip(features.columns, scaler.mean_, scaler.scale_):
            scaling_stats.append(f"{col}, Mean: {mean}, Std: {std}")

    
    with open("scaling_stats.txt", "w") as f:
        f.write("\n".join(scaling_stats))

    print("Saved normalization stats to scaling_stats.txt")
    
    return normalized_features

# 3. Create adjacency matrix based on player distances


def create_player_only_adjacency(df, num_players=28, threshold=None):
    '''
    Fixed version that handles None threshold
    '''
    print(f"Creating player-only adjacency matrix for {num_players} players")
    
    # If threshold is None, use a default value
    if threshold is None:
        threshold = 20.0
        
    # Rest of function is unchanged
    # [Original implementation continues]
def create_distance_adjacency(df, num_nodes=29, threshold=20.0):
    '''
    Create an adjacency matrix for MTGNN graph construction.
    Now improved to focus on player relationships only.
    
    Args:
        df: DataFrame with player and ball data
        num_nodes: Total nodes (29 - 28 players + 1 ball)
        threshold: Distance threshold for connections
    
    Returns:
        Adjacency matrix tensor (compatible with MTGNN)
    '''
    # For MTGNN compatibility, we need a matrix that includes the ball position node
    # But we'll construct the graph based on player relationships only
    
    # Create player-only adjacency (28x28)
    player_adj = create_player_only_adjacency(df, num_players=28, threshold=threshold)
    
    # For compatibility with existing code, expand to include ball node
    # The ball isn't structurally part of the graph but included for compatibility
    full_adj = torch.ones(num_nodes, num_nodes)
    
    # Copy player adjacency to top-left corner
    full_adj[:28, :28] = player_adj
    
    print(f"Final adjacency matrix shape: {full_adj.shape}")
    return full_adj
    
def prepare_sequences(node_features, seq_length=4, pred_length=1):
    """
    Create sequences of data for MTGNN model
    Args:
        node_features: List of dataframes with features for each node
        seq_length: Length of input sequence
        pred_length: How many steps ahead to predict
    """
    sequences = []
    targets_ball_x = []
    targets_ball_y = []
    targets_possession = []
    
    # Ensure all node features have the same length
    min_length = min([len(df) for df in node_features])
    
    # Get feature dimensions for each node to ensure consistent shapes
    feature_dims = [df.shape[1] for df in node_features]
    print(f"Feature dimensions per node: {feature_dims}")
    
    # Number of valid sequences
    num_sequences = min_length - seq_length - pred_length + 1
    print(f"Creating {num_sequences} sequences with length {seq_length}")
    
    for i in range(num_sequences):
        # Prepare input sequence
        seq_data = []
        for node_idx, node_df in enumerate(node_features):
            # Extract sequence for this node
            node_seq = node_df.iloc[i:i+seq_length].values
            seq_data.append(node_seq)
        
        # Instead of converting the entire seq_data to numpy array at once,
        # store it as a list of arrays for now
        sequences.append(seq_data)
        
        # Prepare target values (ball position and possession)
        # Assuming the ball is the last node and its first two features are x, y
        ball_node_idx = len(node_features) - 1
        target_idx = i + seq_length + pred_length - 1
        
        # For ball position - last node's position features
        ball_x = node_features[ball_node_idx].iloc[target_idx, 0]  # x-position
        ball_y = node_features[ball_node_idx].iloc[target_idx, 1]  # y-position
        
        targets_ball_x.append(ball_x)
        targets_ball_y.append(ball_y)
    
    # Convert to appropriate tensor format for MTGNN
    # Format: [batch_size, in_dim (features), num_nodes, seq_len]
    formatted_sequences = []
    
    # We need to ensure all nodes have the same number of features
    max_features = max(feature_dims)
    print(f"Standardizing all nodes to have {max_features} features")
    
    for seq in sequences:
        # Create a padded array for this sequence
        padded_seq = np.zeros((len(seq), max_features, seq_length))
        
        for node_idx, node_seq in enumerate(seq):
            # For each node, fill in the available features
            num_features = node_seq.shape[1]
            padded_seq[node_idx, :num_features, :] = node_seq.T
        
        # Now transpose to get [features, nodes, sequence]
        transposed = np.transpose(padded_seq, (1, 0, 2))
        formatted_sequences.append(transposed)
    
    X = np.array(formatted_sequences)
    y_ball_x = np.array(targets_ball_x)
    y_ball_y = np.array(targets_ball_y)
    
    print(f"Final data shape: {X.shape}")
    
    return X, y_ball_x, y_ball_y

# 5. Split data into train/validation/test sets
def split_data(X, y_ball_x, y_ball_y, train_ratio=0.7, val_ratio=0.15):
    """
    Split data into training, validation and test sets
    """
    # First split into train and temp
    X_train, X_temp, y_ball_x_train, y_ball_x_temp, y_ball_y_train, y_ball_y_temp = train_test_split(
        X, y_ball_x, y_ball_y, test_size=(1 - train_ratio), random_state=42
    )
    
    # Then split temp into validation and test
    test_ratio = 1 - train_ratio - val_ratio
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    
    X_val, X_test, y_ball_x_val, y_ball_x_test, y_ball_y_val, y_ball_y_test = train_test_split(
        X_temp, y_ball_x_temp, y_ball_y_temp, test_size=(1 - val_test_ratio), random_state=42
    )
    
    return (
        X_train, X_val, X_test, 
        y_ball_x_train, y_ball_x_val, y_ball_x_test,
        y_ball_y_train, y_ball_y_val, y_ball_y_test
    )

# 6. Main function to prepare data for MTGNN
def prepare_football_data_for_mtgnn(csv_path, seq_length=200, pred_length=200, use_threshold=False):
    """
    Complete pipeline to prepare football tracking data for MTGNN model
    """
    try:
        # Load and preprocess data
        print("Step 1: Loading and preprocessing data...")
        df = load_and_preprocess_data(csv_path)
        
        # Extract features for each node
        print("Step 2: Extracting features for each node...")
        node_features = extract_features(df)
        
        if not node_features:
            raise ValueError("No node features extracted. Check if player and ball data exists in the dataset.")
        
        # Create adjacency matrix
        print("Step 3: Creating adjacency matrix...")
        threshold = 20 if use_threshold else None  # Arbitrary threshold, adjust as needed
        adj_matrix = create_distance_adjacency(df, len(node_features), threshold)
        
        # Prepare sequences
        print("Step 4: Preparing sequences...")
        X, y_ball_x, y_ball_y = prepare_sequences(node_features, seq_length, pred_length)
        
        # Split data
        print("Step 5: Splitting data into train/val/test sets...")
        splits = split_data(X, y_ball_x, y_ball_y)
        
        return splits, adj_matrix, node_features
        
    except Exception as e:
        print(f"Error in prepare_football_data_for_mtgnn: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
