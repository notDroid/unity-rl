import os
import math
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

class Stopwatch:
    def __init__(self):
        pass

    def start(self):
        self.start_time = time.time()

    def end(self):
        return time.time() - self.start_time

def round_up(x: float | int, y: int = 1):
    return ((x + y - 1) // y) * y

def plot_results(path, logger, log_index=None):
    # Get dataframe
    df = logger.dataframe()
    if log_index:
        df = df.set_index(log_index)
    df.index = pd.to_numeric(df.index, errors='coerce')

    sns.set_theme(style="ticks", font_scale=1.1, palette="tab20")
    rows = (len(df.columns) + 3) // 4
    axes = df.plot(
        subplots=True, 
        layout=(rows, 4), 
        figsize=(20, rows * 3), 
        sharex=False,         
        grid=True,           
        colormap='tab20',  
        legend=True,        
        linewidth=3,
        title="Experiment Results",
    )
    plt.tight_layout()

    # Save
    plt.savefig(path)

def create_animation(imgs):
    fig, ax = plt.subplots()
    ax.axis('off')
    frames = []

    for img in imgs:        
        frame = ax.imshow(img, animated=True)
        frames.append([frame])

    ani = ArtistAnimation(fig, frames, interval=50, blit=True)
    plt.close()
    return ani