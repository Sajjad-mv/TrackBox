# Football Tracking Data - Feature Engineering 

import pandas as pd
import numpy as np
import os
import time

def calculate_speed(df, x_col, y_col):
    return np.sqrt(np.diff(df[x_col], prepend=np.nan)**2 + np.diff(df[y_col], prepend=np.nan)**2) * 0.01 / 0.1

def calculate_acceleration(speed_array):
    return np.diff(speed_array, prepend=np.nan) / 0.1

def calculate_distance_to_ball(df, player_x_col, player_y_col, ball_x_col, ball_y_col):
    if ball_x_col in df.columns and ball_y_col in df.columns:
        return np.sqrt((df[player_x_col] - df[ball_x_col])**2 + (df[player_y_col] - df[ball_y_col])**2) * 0.01
    else:
        return np.full(df.shape[0], np.nan)

def calculate_relative_speed(player_speed_array, ball_speed_array):
    if ball_speed_array is not None:
        return player_speed_array - ball_speed_array
    else:
        return np.full(len(player_speed_array), np.nan)

def calculate_direction(df, x_col, y_col):
    return np.degrees(np.arctan2(np.diff(df[y_col], prepend=np.nan), np.diff(df[x_col], prepend=np.nan)))

def detect_attack_direction(df, player_x_cols, period_col):
    print(" Detecting attacking direction...")

    # Use the first N frames to determine where most home players are positioned
    first_half_frames = df[df[period_col] == 1].head(50)  # First 50 frames
    avg_home_x = first_half_frames[player_x_cols].mean().mean()  # Get avg X position of home players

    home_attacks_right_1st_half = avg_home_x < 0  # If avg x is negative, they defend left & attack right
    home_attacks_right_2nd_half = not home_attacks_right_1st_half  # Flip sides in 2nd half

    print(f" Home team {'attacks right' if home_attacks_right_1st_half else 'attacks left'} in the first half.")
    print(f" Home team {'attacks right' if home_attacks_right_2nd_half else 'attacks left'} in the second half.")

    return {
        'home_attacks_right_1st_half': home_attacks_right_1st_half,
        'home_attacks_right_2nd_half': home_attacks_right_2nd_half
    }


def calculate_zone(df, ball_x_col, period_col, attack_direction):
    print(" Calculating ball zones...")

    midfield_range = 17.5  # Midfield zone range (-17.5m to +17.5m)

    # Identify attack directions per frame
    home_attacking_right = np.where(df[period_col] == 1, 
                                    attack_direction['home_attacks_right_1st_half'], 
                                    attack_direction['home_attacks_right_2nd_half'])
    away_attacking_right = ~home_attacking_right  # Away team attacks in the opposite direction

    zone_df = pd.DataFrame(index=df.index)

    # Midfield Zone
    zone_df['Zone_Midfield'] = np.where((df[ball_x_col] >= -midfield_range) & (df[ball_x_col] <= midfield_range), 1, 0)

    # Attacking and Defensive Zones
    zone_df['Zone_Attacking_Home'] = np.where(
        (home_attacking_right & (df[ball_x_col] > midfield_range)) | (~home_attacking_right & (df[ball_x_col] < -midfield_range)), 
        1, 0
    )
    zone_df['Zone_Defensive_Home'] = np.where(
        (home_attacking_right & (df[ball_x_col] < -midfield_range)) | (~home_attacking_right & (df[ball_x_col] > midfield_range)), 
        1, 0
    )

    zone_df['Zone_Attacking_Away'] = np.where(
        (away_attacking_right & (df[ball_x_col] > midfield_range)) | (~away_attacking_right & (df[ball_x_col] < -midfield_range)), 
        1, 0
    )
    zone_df['Zone_Defensive_Away'] = np.where(
        (away_attacking_right & (df[ball_x_col] < -midfield_range)) | (~away_attacking_right & (df[ball_x_col] > midfield_range)), 
        1, 0
    )

    return zone_df

def calculate_possession_vectorized(df, ball_x_col, ball_y_col, player_columns):
    """
     Improved Possession Detection:
     Assigns possession to the closest player dynamically.
     Uses a refined buffer for smoother transitions.
     Expands the possession threshold for realism.
    """
    print(f" Calculating possession dynamically...")

    if ball_x_col not in df.columns or ball_y_col not in df.columns:
        print(f" Ball position columns missing!")
        return pd.Series(["Loose Ball"] * len(df))

    n_frames = len(df)
    min_distances = np.full(n_frames, np.inf)
    closest_players = np.array(["Loose Ball"] * n_frames, dtype=object)

    ball_x = df[ball_x_col].values
    ball_y = df[ball_y_col].values

    # Step 1: Identify the closest player per frame
    for player in player_columns:
        player_dist_col = f'{player}_DistanceToBall'
        if player_dist_col not in df.columns:
            continue  # Skip if player distance is missing

        player_dist = df[player_dist_col].values
        valid_mask = ~np.isnan(player_dist)

        closer_mask = valid_mask & (player_dist < min_distances)
        min_distances[closer_mask] = player_dist[closer_mask]
        closest_players[closer_mask] = player

    print(f" Closest player assigned dynamically.")

    # Convert distances to meters
    min_distances_m = min_distances * 0.01  # Convert from cm to meters
    possession_threshold = 2  # Increased threshold to 2 meters

    # Step 2: Assign possession if within threshold
    possession_mask = min_distances_m <= possession_threshold
    possession = np.array(["Loose Ball"] * n_frames, dtype=object)
    possession[possession_mask] = closest_players[possession_mask]

    # Step 3: Implement a **smarter buffer** to stabilize possession
    buffer_size = 2  # Reduced from 4 to 2 frames for faster response
    possession_buffer = np.array(["Loose Ball"] * n_frames, dtype=object)
    buffer_count = np.zeros(n_frames, dtype=int)

    for i in range(1, n_frames):
        if possession[i] == possession[i - 1]:  
            buffer_count[i] = min(buffer_count[i - 1] + 1, buffer_size)
        else:
            buffer_count[i] = 1  

        if buffer_count[i] >= buffer_size:
            possession_buffer[i] = possession[i]
        else:
            possession_buffer[i] = possession_buffer[i - 1]

    # Step 4: Assign first valid possession
    first_valid_idx = np.where(possession_buffer != "Loose Ball")[0]
    if len(first_valid_idx) > 0:
        first_valid_possession = possession_buffer[first_valid_idx[0]]
        possession_buffer[:first_valid_idx[0]] = first_valid_possession

    # Step 5: Fill remaining NaN values
    possession_series = pd.Series(possession_buffer)
    possession_series.replace("Loose Ball", np.nan, inplace=True)
    possession_series.ffill(inplace=True)

    print(f" Possession successfully assigned for {possession_series.notna().sum()}/{n_frames} frames.")
    
    return possession_series


def extract_features(input_dir, output_dir, match_ids):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for match_id in match_ids:
        file_path = os.path.join(input_dir, f'{match_id}_Merged.csv')
        print(f"Processing match {match_id}...")
        df = pd.read_csv(file_path)

        # Identify ball columns
        ball_x_col = next((col for col in df.columns if 'ball_x' in col.lower()), None)
        ball_y_col = next((col for col in df.columns if 'ball_y' in col.lower()), None)
        
        print(f"Detected ball columns: {ball_x_col}, {ball_y_col}")

        # Identify player columns
        player_columns = []
        for col in df.columns:
            if '_x' in col and 'ball' not in col.lower():
                player_id = col[:-2]  # Remove '_x'
                if f"{player_id}_y" in df.columns:  # Verify y column exists
                    player_columns.append(player_id)
        
        player_columns = list(set(player_columns))
        print(f"Detected {len(player_columns)} players")

        player_x_cols = [col for col in df.columns if '_x' in col and 'home' in col]  # Home team x-coordinates
        attack_direction = detect_attack_direction(df, player_x_cols, 'IdPeriod')

        additional_features = {}

        if ball_x_col and ball_y_col:
            print("Calculating ball features...")
            additional_features['Ball_Speed'] = calculate_speed(df, ball_x_col, ball_y_col)
            additional_features['Ball_Acceleration'] = calculate_acceleration(additional_features['Ball_Speed'])
            additional_features['Ball_Direction'] = calculate_direction(df, ball_x_col, ball_y_col)
            
            # Add zone features
            print(" Calculating zone features...")
            zone_features = calculate_zone(df, 'ball_x_Home', 'IdPeriod', attack_direction)

            # Step 3: Add to additional features
            additional_features.update(zone_features)

        #  **Step 1: Compute Player Distances Before Possession**
        print("Calculating player features...")
        player_data = []
        
        for i, player in enumerate(player_columns):
            if i % 10 == 0:
                print(f"Processing player features {i+1}/{len(player_columns)}")
                
            player_x_col = f'{player}_x'
            player_y_col = f'{player}_y'
            
            player_present_mask = df[player_x_col].notna() & df[player_y_col].notna()
            
            player_speed = np.where(player_present_mask, calculate_speed(df, player_x_col, player_y_col), np.nan)
            player_acceleration = np.where(player_present_mask, calculate_acceleration(player_speed), np.nan)
            player_direction = np.where(player_present_mask, calculate_direction(df, player_x_col, player_y_col), np.nan)
            
            player_distance_to_ball = np.where(
                player_present_mask, 
                calculate_distance_to_ball(df, player_x_col, player_y_col, ball_x_col, ball_y_col), 
                np.nan
            )
            
            # Only calculate relative speed if ball speed is available
            player_relative_speed = np.full(len(df), np.nan)
            if 'Ball_Speed' in additional_features:
                player_relative_speed = np.where(
                    player_present_mask,
                    calculate_relative_speed(player_speed, additional_features.get('Ball_Speed')),
                    np.nan
                )
            
            player_data.append(pd.DataFrame({
                f'{player}_Speed': player_speed,
                f'{player}_Acceleration': player_acceleration,
                f'{player}_Direction': player_direction,
                f'{player}_DistanceToBall': player_distance_to_ball,  # ✅ Fix: Ensure distances exist
                f'{player}_RelativeSpeedToBall': player_relative_speed
            }))

        #  **Step 2: Store Player Distances Before Running Possession**
        print("Combining all features...")
        df_final = pd.concat([df] + player_data + [pd.DataFrame(additional_features)], axis=1)

        #  **Step 3: Run Possession Calculation After Distances Are Available**
        print("Calculating possession...")
        df_final["Possession"] = calculate_possession_vectorized(
            df_final, ball_x_col, ball_y_col, player_columns
        )

        # Save to output file
        output_file = os.path.join(output_dir, f'{match_id}_Features.csv')
        df_final.to_csv(output_file, index=False)
        print(f" Features extracted and saved: {output_file}")


if __name__ == '__main__':
    input_dir = '/project_ghent/Test/Assignment/Merged'
    output_dir = '/project_ghent/Test/Assignment/Features'
    match_ids = ['match_1', 'match_2', 'match_3', 'match_4', 'match_5']

    extract_features(input_dir, output_dir, match_ids)


# Description:
# - Calculates player and ball speed (m/s) using coordinates in cm and 1/10 second intervals.
# - Calculates acceleration (m/s²) as the rate of change of speed.
# - Calculates the distance between each player and the ball (m).
# - Determines possession based on the player closest to the ball.
# - Saves the extracted features in separate CSV files for each match.
