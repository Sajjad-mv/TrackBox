# simple_norm_fix.py - Direct implementation without generating other files
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

def load_and_preprocess_data(csv_path):
    """Load and preprocess data"""
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=['IdPeriod', 'Time'])
    return df

def extract_features_fixed(df, test_mode=False, saved_scalers=None):
    """Extract features with proper normalization handling"""
    node_features = []
    scalers = [] if saved_scalers is None else saved_scalers
    
    # Extract home players
    for i in range(1, 15):
        if f'home_{i}_x' in df.columns and f'home_{i}_y' in df.columns:
            player_features = df[[
                f'home_{i}_x', 
                f'home_{i}_y',
                f'home_{i}_DistanceToBall'
            ]]
            
            # Handle normalization properly
            if test_mode and saved_scalers:
                # Use saved scaler for test data
                scaler_idx = len(node_features)
                if scaler_idx < len(saved_scalers):
                    scaler = saved_scalers[scaler_idx]
                    normalized = pd.DataFrame(
                        scaler.transform(player_features.fillna(0)), 
                        columns=player_features.columns
                    )
                else:
                    # Fallback if scaler missing
                    normalized = player_features.fillna(0)
            else:
                # Create new scaler for training data
                scaler = StandardScaler()
                normalized = pd.DataFrame(
                    scaler.fit_transform(player_features.fillna(0)), 
                    columns=player_features.columns
                )
                scalers.append(scaler)
            
            node_features.append(normalized)
    
    # Extract away players (same pattern)
    for i in range(1, 15):
        if f'away_{i}_x' in df.columns and f'away_{i}_y' in df.columns:
            player_features = df[[
                f'away_{i}_x', 
                f'away_{i}_y',
                f'away_{i}_DistanceToBall'
            ]]
            
            # Handle normalization properly
            if test_mode and saved_scalers:
                # Use saved scaler for test data
                scaler_idx = len(node_features)
                if scaler_idx < len(saved_scalers):
                    scaler = saved_scalers[scaler_idx]
                    normalized = pd.DataFrame(
                        scaler.transform(player_features.fillna(0)), 
                        columns=player_features.columns
                    )
                else:
                    # Fallback if scaler missing
                    normalized = player_features.fillna(0)
            else:
                # Create new scaler for training data
                scaler = StandardScaler()
                normalized = pd.DataFrame(
                    scaler.fit_transform(player_features.fillna(0)), 
                    columns=player_features.columns
                )
                scalers.append(scaler)
            
            node_features.append(normalized)
    
    # Extract ball features
    ball_columns = ['ball_x_Home', 'ball_y_Home', 'Ball_Speed', 'Ball_Acceleration', 'Ball_Direction_Degrees']
    ball_columns = [col for col in ball_columns if col in df.columns]
    
    if ball_columns:
        ball_features = df[ball_columns]
        
        # Handle normalization properly
        if test_mode and saved_scalers:
            # Use saved scaler for test data
            scaler_idx = len(node_features)
            if scaler_idx < len(saved_scalers):
                scaler = saved_scalers[scaler_idx]
                normalized = pd.DataFrame(
                    scaler.transform(ball_features.fillna(0)), 
                    columns=ball_features.columns
                )
            else:
                # Fallback if scaler missing
                normalized = ball_features.fillna(0)
        else:
            # Create new scaler for training data
            scaler = StandardScaler()
            normalized = pd.DataFrame(
                scaler.fit_transform(ball_features.fillna(0)), 
                columns=ball_features.columns
            )
            scalers.append(scaler)
        
        node_features.append(normalized)
        
        # Special handling for prediction target (ball x,y)
        # Save ball position scaler separately for denormalization
        if not test_mode:
            ball_pos_scaler = StandardScaler()
            ball_pos_scaler.fit(df[['ball_x_Home', 'ball_y_Home']].fillna(0))
            scalers.append(('ball_pos', ball_pos_scaler))
    
    return node_features, scalers

def prepare_sequences(node_features, seq_length=4):
    """
    Create sequences of data for MTGNN model
    """
    sequences = []
    targets_ball_x = []
    targets_ball_y = []
    
    # Ensure all node features have the same length
    min_length = min([len(df) for df in node_features])
    
    # Get feature dimensions for each node to ensure consistent shapes
    feature_dims = [df.shape[1] for df in node_features]
    print(f"Feature dimensions per node: {feature_dims}")
    
    # Number of valid sequences
    num_sequences = min_length - seq_length - 1 + 1  # pred_length=1
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
        target_idx = i + seq_length  # prediction 1 step ahead
        
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
        transposed = np.transpose(padded_seq, (2, 0, 1))
        formatted_sequences.append(transposed)
    
    X = np.array(formatted_sequences)
    X = np.transpose(X, (0, 3, 2, 1))
    y_ball_x = np.array(targets_ball_x)
    y_ball_y = np.array(targets_ball_y)
    
    print(f"Final data shape: {X.shape}")
    
    return X, y_ball_x, y_ball_y

def prepare_football_data_fixed(csv_path, seq_length=4, train_ratio=0.8):
    """
    Pipeline with proper normalization handling
    """
    # Load data
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=['IdPeriod', 'Time'])
    
    # First split into train and test sets - BEFORE normalization
    train_size = int(len(df) * train_ratio)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    
    print(f"Split data: train={len(df_train)} rows, test={len(df_test)} rows")
    
    # Process training data and save scalers
    train_features, scalers = extract_features_fixed(df_train)
    
    # Save scalers for later use
    torch.save(scalers, "feature_scalers.pt")
    print(f"Saved {len(scalers)} scalers for denormalization")
    
    # Extract ball position scalers for target denormalization
    ball_pos_scaler = None
    for item in scalers:
        if isinstance(item, tuple) and item[0] == 'ball_pos':
            ball_pos_scaler = item[1]
            break
    
    # If no special ball scaler found, use the ball node scaler
    if ball_pos_scaler is None and len(scalers) >= 29:  # 28 players + ball
        ball_pos_scaler = scalers[28]  # Ball should be the last node
        
    # Save key normalization parameters for ball position
    if ball_pos_scaler:
        with open("ball_pos_norm_params.txt", "w") as f:
            f.write(f"ball_x_mean: {ball_pos_scaler.mean_[0]}\n")
            f.write(f"ball_x_std: {ball_pos_scaler.scale_[0]}\n")
            f.write(f"ball_y_mean: {ball_pos_scaler.mean_[1]}\n")
            f.write(f"ball_y_std: {ball_pos_scaler.scale_[1]}\n")
        print("Saved ball position normalization parameters")
    
    # Process test data using same scalers
    test_features, _ = extract_features_fixed(df_test, test_mode=True, saved_scalers=scalers)
    
    # Prepare sequences for training data
    X_train, y_ball_x_train, y_ball_y_train = prepare_sequences(train_features, seq_length)
    
    # Prepare sequences for test data
    X_test, y_ball_x_test, y_ball_y_test = prepare_sequences(test_features, seq_length)
    
    # Create basic validation set from training data
    X_train, X_val, y_ball_x_train, y_ball_x_val, y_ball_y_train, y_ball_y_val = train_test_split(
        X_train, y_ball_x_train, y_ball_y_train, test_size=0.15, random_state=42
    )
    
    # Save test data for evaluation
    torch.save(X_test, "test_data.pt")
    torch.save((y_ball_x_test, y_ball_y_test), "test_targets.pt")
    
    # Return everything needed
    return (
        (X_train, X_val, X_test, y_ball_x_train, y_ball_x_val, y_ball_x_test, y_ball_y_train, y_ball_y_val, y_ball_y_test),
        scalers
    )

def denormalize_predictions(predictions, norm_params_file=None):
    """Denormalize predictions using saved parameters"""
    # Try to load saved scalers first
    if os.path.exists("feature_scalers.pt"):
        try:
            scalers = torch.load("feature_scalers.pt")
            # Find ball position scaler
            ball_pos_scaler = None
            for item in scalers:
                if isinstance(item, tuple) and item[0] == 'ball_pos':
                    ball_pos_scaler = item[1]
                    break
            
            # If found, use it
            if ball_pos_scaler:
                print("Using saved ball position scaler")
                return ball_pos_scaler.inverse_transform(predictions)
        except:
            print("Could not use saved scalers")
    
    # Fallback to manual parameters
    if norm_params_file and os.path.exists(norm_params_file):
        try:
            # Read parameters
            params = {}
            with open(norm_params_file, 'r') as f:
                for line in f:
                    key, value = line.strip().split(': ')
                    params[key] = float(value)
            
            # Apply denormalization
            x_mean = params.get('ball_x_mean', 0)
            x_std = params.get('ball_x_std', 1)
            y_mean = params.get('ball_y_mean', 0)
            y_std = params.get('ball_y_std', 1)
            
            denorm_predictions = predictions.copy()
            denorm_predictions[:, 0] = predictions[:, 0] * x_std + x_mean  # x
            denorm_predictions[:, 1] = predictions[:, 1] * y_std + y_mean  # y
            
            print(f"Denormalized using parameters from {norm_params_file}")
            return denorm_predictions
        except Exception as e:
            print(f"Error using norm params file: {str(e)}")
    
    print("WARNING: Could not denormalize predictions")
    return predictions

def train_model():
    """Train model with proper normalization"""
    from model_training import run_football_prediction_pipeline
    
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Prepare data with proper normalization
    print("Preparing data with proper normalization...")
    csv_path = 'match_1_Features.csv'
    data_splits, scalers = prepare_football_data_fixed(csv_path, seq_length=4)
    
    # Create fixed adjacency matrix 
    print("Creating adjacency matrix...")
    num_nodes = data_splits[0].shape[2]  # X_train
    adj_matrix = torch.ones(num_nodes, num_nodes)
    
    # Train model
    print("Training model...")
    batch_size = 32
    epochs = 20
    
    model, results, metrics = run_football_prediction_pipeline(
        data_splits,
        adj_matrix=adj_matrix,
        node_features=None,  # Not needed in this version
        batch_size=batch_size,
        epochs=epochs,
        device=device
    )
    
    # Save model
    torch.save(model.state_dict(), "fixed_trained_model.pth")
    print("Model saved to fixed_trained_model.pth")
    
    print("Training complete!")

def evaluate_model():
    """Evaluate model with proper denormalization and NaN handling"""
    from model_training import initialize_mtgnn_model
    import numpy as np
    
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    if not os.path.exists("test_data.pt") or not os.path.exists("test_targets.pt"):
        print("Test data not found. Please run training first.")
        return
    
    X_test = torch.load("test_data.pt")
    y_ball_x_test, y_ball_y_test = torch.load("test_targets.pt")
    
    # Load model
    model_path = "fixed_trained_model.pth"
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found")
        return
    
    # Create model with same architecture
    _, in_dim, num_nodes, seq_length = X_test.shape
    model = initialize_mtgnn_model(num_nodes, in_dim, seq_length, device)
    model.load_state_dict(torch.load(model_path))
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        # Convert to tensor
        test_tensor = torch.FloatTensor(X_test).to(device)
        
        # Create adjacency matrix
        adj_matrix = torch.ones(num_nodes, num_nodes).to(device)
        
        # Forward pass
        outputs = model(test_tensor, A_tilde=adj_matrix)
        
        # Extract ball node prediction
        ball_outputs = outputs[:, :, -1, :].mean(dim=-1)  # [batch_size, out_dim]
        predictions = ball_outputs.cpu().numpy()
    
    # Prepare actual targets
    targets = np.column_stack((y_ball_x_test, y_ball_y_test))
    
    # Calculate normalized metrics
    mse_norm = mean_squared_error(targets, predictions)
    mae_norm = mean_absolute_error(targets, predictions)
    
    print(f"Normalized metrics:")
    print(f"MSE: {mse_norm:.6f}")
    print(f"MAE: {mae_norm:.6f}")
    
    # Denormalize predictions
    denorm_predictions = denormalize_predictions(predictions, "ball_pos_norm_params.txt")
    
    # Load original data for comparison
    csv_path = "match_1_Features.csv"
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=['IdPeriod', 'Time'])
    
    # Get test portion
    train_ratio = 0.8
    train_size = int(len(df) * train_ratio)
    df_test = df.iloc[train_size:]
    
    # Extract actual ball positions (accounting for sequence length)
    seq_length = 4
    ball_positions = df_test[['ball_x_Home', 'ball_y_Home']].values[seq_length:seq_length+len(predictions)]
    
    # Check for NaN values in both arrays and create masks
    print(f"Checking for NaN values:")
    has_nan_ball = np.isnan(ball_positions).any()
    has_nan_pred = np.isnan(denorm_predictions).any()
    print(f"NaN in ball_positions: {has_nan_ball}")
    print(f"NaN in denorm_predictions: {has_nan_pred}")
    
    # Create valid mask (where both arrays have good values)
    valid_indices = ~np.isnan(ball_positions).any(axis=1) & ~np.isnan(denorm_predictions).any(axis=1)
    
    # Filter out NaN values for comparison
    ball_positions_valid = ball_positions[valid_indices]
    denorm_predictions_valid = denorm_predictions[valid_indices]
    
    print(f"Original data points: {len(ball_positions)}")
    print(f"Valid data points for comparison: {len(ball_positions_valid)}")
    
    # Compare with original (only on valid indices)
    if len(ball_positions_valid) > 0:
        mse_original = mean_squared_error(ball_positions_valid, denorm_predictions_valid)
        mae_original = mean_absolute_error(ball_positions_valid, denorm_predictions_valid)
        
        print(f"Denormalized metrics:")
        print(f"MSE: {mse_original:.6f}")
        print(f"MAE: {mae_original:.6f}")
        
        # Visualize
        plt.figure(figsize=(12, 10))
        
        # Limit to 100 points for clarity
        display_limit = min(100, len(ball_positions_valid))
        
        # Plot actual positions
        plt.plot(ball_positions_valid[:display_limit, 0], ball_positions_valid[:display_limit, 1], 
                'b-', label='Actual Path')
        plt.scatter(ball_positions_valid[:display_limit, 0], ball_positions_valid[:display_limit, 1], 
                    c='blue', alpha=0.6, label='Actual Positions')
        
        # Plot predicted positions
        plt.plot(denorm_predictions_valid[:display_limit, 0], denorm_predictions_valid[:display_limit, 1], 
                'r--', label='Predicted Path')
        plt.scatter(denorm_predictions_valid[:display_limit, 0], denorm_predictions_valid[:display_limit, 1], 
                    c='red', alpha=0.6, label='Predicted Positions')
        
        plt.title(f'Ball Position: Actual vs Predicted (First {display_limit} points)')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig('ball_position_comparison.png')
        print("Saved comparison visualization to ball_position_comparison.png")
    else:
        print("ERROR: No valid data points for comparison after filtering NaN values")
        print("Possible solutions:")
        print("1. Check the ball position data in the original CSV")
        print("2. Verify the denormalization process is working correctly")
        print("3. Try using a different normalization approach")

def main():
    """Main function"""
    print("=== Football Data Normalization Fix ===")
    print("This script fixes normalization issues in the football prediction model")
    print("\nWhat would you like to do?")
    print("1. Train model with fixed normalization")
    print("2. Evaluate model with proper denormalization")
    print("3. Both train and evaluate")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == '1':
        train_model()
    elif choice == '2':
        evaluate_model()
    elif choice == '3':
        train_model()
        evaluate_model()
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
