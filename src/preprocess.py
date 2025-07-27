from statsbombpy import sb
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Settings: Champions League 2017/2018 (comp_id=16, season_id=1)
COMPETITION_ID = 16
SEASON_ID = 1

# Output directory
output_dir = "input/_dataset/train_test_data_final"
os.makedirs(output_dir, exist_ok=True)

# Fetch matches
matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
print(f"Number of matches found: {len(matches)}")

all_shots = []

for match_id in matches['match_id']:
    print(f"Processing match {match_id}")
    events = sb.events(match_id=match_id)

    # Filter shot events
    shots = events[events['type'] == 'Shot'].copy()

    # Select relevant columns for xG modeling
    shots = shots[[
        'match_id',
        'period',
        'timestamp',
        'team',
        'player',
        'location',
        'shot_outcome',
        'shot_body_part',
        'shot_type',
        'shot_statsbomb_xg'
    ]]

    all_shots.append(shots)

# Concatenate all shots into one dataframe
shots_df = pd.concat(all_shots, ignore_index=True)
print(f"Total shots collected: {len(shots_df)}")
# Concatenate all shots into one dataframe
shots_df = pd.concat(all_shots, ignore_index=True)
print(f"Total shots collected: {len(shots_df)}")

# === ADD FEATURE ENGINEERING HERE ===

import math

def calculate_distance(loc):
    x, y = loc
    goal_x, goal_y = 120, 40
    return math.sqrt((goal_x - x)**2 + (goal_y - y)**2)

def calculate_angle(loc):
    x, y = loc
    goal_left = (120, 36)
    goal_right = (120, 44)

    def angle_between_points(p1, p2, p3):
        a = math.dist(p2, p3)
        b = math.dist(p1, p2)
        c = math.dist(p1, p3)
        try:
            angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c))
            return angle
        except:
            return 0

    return angle_between_points((x, y), goal_left, goal_right)

shots_df['distance'] = shots_df['location'].apply(calculate_distance)
shots_df['angle'] = shots_df['location'].apply(calculate_angle)

# Dummy values for now
shots_df['player_in_between'] = 0
shots_df['goal_keeper_angle'] = 0

# === FEATURE ENGINEERING DONE ===

# Add your 'target' column if your model expects it
shots_df['target'] = shots_df['shot_outcome'].apply(lambda x: 1 if x == 'Goal' else 0)

# Split data into train and test sets (80% train, 20% test)
train_df, test_df = train_test_split(shots_df, test_size=0.2, random_state=42)

# Save pickle files required by the training script
train_path = os.path.join(output_dir, "train_label_final.pkl")
test_path = os.path.join(output_dir, "test_label_final.pkl")

train_df.to_pickle(train_path)
test_df.to_pickle(test_path)

print(f"Training data saved to: {train_path}")
print(f"Test data saved to: {test_path}")
