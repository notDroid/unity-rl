import os
import math
import time
import shutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

from huggingface_hub import HfApi, hf_hub_download, upload_file

class Stopwatch:
    def __init__(self):
        pass

    def start(self):
        self.start_time = time.time()

    def end(self):
        return time.time() - self.start_time

def round_up(x: float | int, y: int = 1):
    return ((x + y - 1) // y) * y

def plot_results(logger, path=None, log_index=None, ewma_span=0):
    # Get dataframe
    df = logger.dataframe()
    if log_index:
        df = df.set_index(log_index)
    df.index = pd.to_numeric(df.index, errors='coerce')
    if ewma_span > 0: df_ewma = df.ewm(span=ewma_span).mean()

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
        linewidth=2 if ewma_span>0 else 3,
        title="Experiment Results",
        alpha=0.25 if ewma_span>0 else 1.0
    )
    plt.tight_layout()

    # EWMA
    if ewma_span > 0:
        for col, ax in zip(df.columns, axes.flatten()):
            plot_color = ax.get_lines()[0].get_color()
            ax.plot(df.index, df_ewma[col], color=plot_color, linewidth=3)

            # Align y with ewma
            default_margin = plt.rcParams['axes.ymargin']
            ewma_min = df_ewma[col].min()
            ewma_max = df_ewma[col].max()
            data_range = ewma_max - ewma_min
            if data_range == 0: padding = 1.0 
            else: padding = data_range * default_margin
            ax.set_ylim(ewma_min - padding, ewma_max + padding)

    # Save
    if path: plt.savefig(path)

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

def download_from_hf_hub(local_path, repo_id, remote_path, token=None, **kwargs):
    cached_path = hf_hub_download(
        repo_id=repo_id,
        filename=remote_path,
        token=token,
        repo_type="model",
        **kwargs
    )
    
    local_dir = os.path.dirname(local_path)
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)
    
    shutil.copy2(cached_path, local_path)

def upload_to_hf_hub(local_path, repo_id, remote_path, token=None, **kwargs):
    upload_file(
        path_or_fileobj=local_path,
        path_in_repo=remote_path,
        repo_id=repo_id,
        token=token,
        repo_type="model",
        **kwargs
    )