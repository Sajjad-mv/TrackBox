import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Arc
from matplotlib.widgets import Button

# Load the processed dataset (1 match)
df = pd.read_csv("/project_ghent/Test/Assignment/Features_final/match_1_Features.csv")

# Convert coordinates from cm to meters
df[['ball_x_Home', 'ball_y_Home']] *= 0.01
for col in df.columns:
    if col.endswith('_x') or col.endswith('_y'):
        df[col] *= 0.01

# Pitch dimensions (FIFA standard: 105m x 68m)
pitch_length, pitch_width = 105, 68

# Get player IDs
home_players = [col[5:-2] for col in df.columns if col.startswith('home_') and col.endswith('_x')]
away_players = [col[5:-2] for col in df.columns if col.startswith('away_') and col.endswith('_x')]

# Convert data to NumPy arrays for faster access
home_x_array = df[[f'home_{pid}_x' for pid in home_players]].to_numpy()
home_y_array = df[[f'home_{pid}_y' for pid in home_players]].to_numpy()
away_x_array = df[[f'away_{pid}_x' for pid in away_players]].to_numpy()
away_y_array = df[[f'away_{pid}_y' for pid in away_players]].to_numpy()
ball_x_array = df['ball_x_Home'].to_numpy()
ball_y_array = df['ball_y_Home'].to_numpy()
possession_array = df['Possession'].astype(str).fillna("").to_numpy()

# Setup figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-pitch_length / 2, pitch_length / 2)
ax.set_ylim(-pitch_width / 2, pitch_width / 2)
ax.set_title("Football Tracking Animation")

# Function to draw the pitch
def draw_pitch(ax):
    ax.add_patch(Rectangle((-pitch_length / 2, -pitch_width / 2), pitch_length, pitch_width,
                           linewidth=2, edgecolor="black", facecolor="none"))

    # Center circle
    ax.add_patch(Circle((0, 0), 9.15, edgecolor="black", facecolor="none", linewidth=2))

    # Halfway line
    ax.axvline(0, color='black', linewidth=2)

    # Goal & penalty areas
    for x in [-pitch_length / 2, pitch_length / 2 - 5.5]:
        ax.add_patch(Rectangle((x, -9.16), 5.5, 18.32, edgecolor="black", facecolor="none", linewidth=2))
    for x in [-pitch_length / 2, pitch_length / 2 - 16.5]:
        ax.add_patch(Rectangle((x, -20.16), 16.5, 40.32, edgecolor="black", facecolor="none", linewidth=2))

    # Penalty arcs
    ax.add_patch(Arc((-pitch_length / 2 + 11, 0), 18.3, 18.3, angle=0, theta1=308, theta2=52, edgecolor="black", linewidth=2))
    ax.add_patch(Arc((pitch_length / 2 - 11, 0), 18.3, 18.3, angle=0, theta1=128, theta2=232, edgecolor="black", linewidth=2))

    # Center dot
    ax.add_patch(Circle((0, 0), 0.2, edgecolor="black", facecolor="black"))

# Draw pitch
draw_pitch(ax)

# Initialize scatter plots
home_scatter = ax.scatter([], [], c='blue', label="Home Team", edgecolors='black')
away_scatter = ax.scatter([], [], c='red', label="Away Team", edgecolors='black')
ball_scatter = ax.scatter([], [], c='gold', s=100, label="Ball", edgecolors='black')

# Text annotation for possession
possession_text = ax.text(0, pitch_width / 2 - 2, "", fontsize=12, ha='center', color='black')

# Animation control variables
paused = False
current_frame = 0
frame_skip = 5  # Render every 5th frame for speed

# Function to update animation
def update(frame):
    global current_frame
    if not paused:
        current_frame = frame

    ax.set_title(f"Time: {current_frame / 10:.1f} seconds")  

    # Use NumPy arrays for faster access
    home_scatter.set_offsets(np.c_[home_x_array[current_frame], home_y_array[current_frame]])
    away_scatter.set_offsets(np.c_[away_x_array[current_frame], away_y_array[current_frame]])
    ball_scatter.set_offsets([ball_x_array[current_frame], ball_y_array[current_frame]])

    # Handle possession highlighting efficiently
    possession = possession_array[current_frame]
    if possession:
        color = 'lime' if possession.startswith('home') else 'orange'
        player_id = possession.split('_')[1]  
        if possession.startswith('home'):
            idx = home_players.index(player_id)
            home_scatter.set_facecolors(['lime' if i == idx else 'blue' for i in range(len(home_players))])
        else:
            idx = away_players.index(player_id)
            away_scatter.set_facecolors(['orange' if i == idx else 'red' for i in range(len(away_players))])
        possession_text.set_text(f"Possession: {possession}")
    else:
        home_scatter.set_facecolors('blue')
        away_scatter.set_facecolors('red')
        possession_text.set_text("Possession: None")

    return home_scatter, away_scatter, ball_scatter, possession_text

# Play/Pause button function
def toggle_pause(event):
    global paused
    paused = not paused

# Forward button function (step forward)
def step_forward(event):
    global current_frame
    current_frame = min(current_frame + 1, len(df) - 1)
    update(current_frame)
    plt.draw()

# Create animation
ani = animation.FuncAnimation(fig, update, frames=range(0, len(df), frame_skip), interval=50, blit=True)

# Add interactive buttons
ax_pause = plt.axes([0.7, 0.02, 0.1, 0.05])
ax_forward = plt.axes([0.81, 0.02, 0.1, 0.05])

btn_pause = Button(ax_pause, "Play/Pause")
btn_forward = Button(ax_forward, "Forward")

btn_pause.on_clicked(toggle_pause)
btn_forward.on_clicked(step_forward)

# Save animation as an MP4 file (optimized settings)
save_path = "football_animation2.mp4"
writer = animation.FFMpegWriter(fps=5, metadata={"title": "Football Animation"})
ani.save(save_path, writer=writer)
print(f"Animation saved as {save_path}")

plt.legend()
plt.show()
