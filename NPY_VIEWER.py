import numpy as np
import os
import matplotlib.pyplot as plt

def view_npy_file(file_path):
    data = np.load(file_path)
    
    print(f"File: {file_path}")
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    
    if data.ndim == 1:
        print("Contents:")
        print(data)
    elif data.ndim == 2:
        print("2D array (e.g., image). Displaying first few rows and columns:")
        print(data[:5, :5])
        
        # Optionally, display the image
        plt.imshow(data, cmap='gray')
        plt.title(f"Image from {os.path.basename(file_path)}")
        plt.axis('off')
        plt.show()
    else:
        print("Multi-dimensional array. Displaying first few elements:")
        print(data.flatten()[:20])

# Directory containing the .npy files
npy_dir = '/Users/shrutibalaji/Downloads/vindr-mammo-master/preprocessed_data'

# List all .npy files in the directory
npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]

# View contents of each file
for file in npy_files:
    view_npy_file(os.path.join(npy_dir, file))
    print("\n" + "-"*50 + "\n")