import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import imageio
from matplotlib import transforms
from PIL import Image, ImageOps
from tqdm import tqdm
video = "vid220" #Name of the video
output_dir = f'./trajectories/{video}' #Where .txt files should be
os.makedirs('./working_dir/gifs', exist_ok=True) #Create dir for gif creation
files = sorted(glob.glob(os.path.join(output_dir, '*.txt')))
print(f"processing files from {files}")
missing = []
DEFAULT_OFFSETS = [0, 93, 165]
RANGE = [1350, 1870]

# Read and concatenate all data into a list of DataFrames
dataframes = []
for csv, color in zip(files, ['r', 'g', 'b']):
    df = pd.read_csv(csv, names=['filename', 'x', 'y', 'z', 'alpha', 'beta', 'gamma', 'center_x', 'center_y', 'rvec0', 'rvec1', 'rvec2', 'error', 'corners'], na_values=missing)
    df['color'] = color  # Add color column to differentiate datasets
    dataframes.append(df)

def rotate_image(image_path, color):
    img = Image.open(image_path)
    img = ImageOps.expand(img,border=25,fill=color)
    return img.rotate(-90, expand=True)

def update_plots(offset1=0, offset2=0, offset3=0, end_idx=len(dataframes[0]), save_path=None):
    df1 = dataframes[0].copy()
    df2 = dataframes[1].copy()
    df3 = dataframes[2].copy()
    
    df1['adjusted_index'] = df1.index + offset1
    df1['filename_debug'] = f'{video}_cam1_debug_'+df1['filename']
    df2['adjusted_index'] = df2.index + offset2
    df2['filename_debug'] = f'{video}_cam2_debug_'+df2['filename']
    df3['adjusted_index'] = df3.index + offset3
    df3['filename_debug'] = f'{video}_cam3_debug_'+df3['filename']
    df_all = pd.concat([df1, df2, df3], ignore_index=True)
    df_all = df_all[(df_all['adjusted_index'] > RANGE[0]) & (df_all['adjusted_index'] <= end_idx)]#end_idx
    fig = plt.figure(figsize=(14, 18))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.1, 0.1, 0.2])

    # Plot x
    ax1 = fig.add_subplot(gs[0, 0])
    for color in ['r', 'g', 'b']:
        subset = df_all[df_all['color'] == color]
        ax1.scatter(subset['adjusted_index'], subset['x'], c=color, label=color)
    ax1.set_ylabel('X [m]')
    ax1.set_title('Axis - X')
    
    # Plot y
    ax2 = fig.add_subplot(gs[0, 1])
    for color in ['r', 'g', 'b']:
        subset = df_all[df_all['color'] == color]
        ax2.scatter(subset['adjusted_index'], subset['y'], c=color, label=color)
    ax2.set_ylabel('Y [m]')
    ax2.set_title('Axis Y')
    
    # Plot z
    ax3 = fig.add_subplot(gs[0, 2])
    for color in ['r', 'g', 'b']:
        subset = df_all[df_all['color'] == color]
        ax3.scatter(subset['adjusted_index'], subset['z'], c=color, label=color)
    ax3.set_ylabel('Z [m]')
    ax3.set_title('Axis - Z')
    
    # Plot Psi
    ax4 = fig.add_subplot(gs[1, 0])
    for color in ['r', 'g', 'b']:
        subset = df_all[df_all['color'] == color]
        ax4.scatter(subset['adjusted_index'], subset['alpha'], c=color, label=color)
    ax4.set_ylabel('Psi [Degree]')
    ax4.set_title('Axis - Psi')
    
    # Plot Theta
    ax5 = fig.add_subplot(gs[1, 1])
    for color in ['r', 'g', 'b']:
        subset = df_all[df_all['color'] == color]
        ax5.scatter(subset['adjusted_index'], subset['beta'], c=color, label=color)
    ax5.set_ylabel('Theta [Degree]')
    ax5.set_title('Axis - Theta')
    
    # Plot Phi
    ax6 = fig.add_subplot(gs[1, 2])
    for color in ['r', 'g', 'b']:
        subset = df_all[df_all['color'] == color]
        ax6.scatter(subset['adjusted_index'], subset['gamma'], c=color, label=color)
    ax6.set_ylabel('Phi [Degree]')
    ax6.set_title('Axis - Phi')
    
    # Plot 3D scatter
    for i, color, image in zip(range(3), ['red','green','blue'], df_all[df_all['adjusted_index']==end_idx]['filename_debug']): #end_idx
        ax = fig.add_subplot(gs[2, i])
        img = rotate_image(f"./debug/{video}/{image}", color)
        ax.imshow(img)
        
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

# Create a list to store file paths for the frames
frames = []
# Generate frames
for i in tqdm(range(RANGE[0], RANGE[1])):
    frame_path = f"./working_dir/gifs/frame_{i:04d}.png"
    update_plots(offset1=DEFAULT_OFFSETS[0], offset2=DEFAULT_OFFSETS[1], offset3=DEFAULT_OFFSETS[2], end_idx=i, save_path=frame_path)
    frames.append(frame_path)

# Create the GIF
with imageio.get_writer(f'{output_dir}/animated_plot.gif', mode='I', duration=0.1) as writer:
    for frame in frames:
        image = imageio.imread(frame)
        writer.append_data(image)

# Clean up frames
for frame in frames:
    os.remove(frame)
