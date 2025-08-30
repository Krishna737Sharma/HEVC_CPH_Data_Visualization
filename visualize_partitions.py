import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Configuration ---

# Path to your dataset file
FILE_PATH = '/workspaces/HEVC_CPH_Data_Visualization/AI_Valid_143925.dat'

# The index of the sample you want to see (e.g., 0, 50, 100)
SAMPLE_INDEX = 100

# The Quantization Parameter (QP) you want to inspect (0-51)
QP_TO_INSPECT = 37

# --- Data Structure Constants ---
IMAGE_SIZE = 64
NUM_SAMPLE_LENGTH = 4992
IMAGE_BYTES = 4096
NUM_LABEL_BYTES = 16

def draw_cu_partitions(ax, label_data):
    """
    Draws the CU partition grid on the image based on the 16-byte label.
    """
    # If all labels are 0, the whole 64x64 block is one partition. Do nothing.
    if np.all(label_data == 0):
        return

    # Helper function to draw a rectangle
    def draw_rect(x, y, size, color='red'):
        rect = patches.Rectangle((x, y), size, size, linewidth=1.2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    # Check for 32x32 splits
    # A 32x32 block is NOT split if all its 4 corresponding labels are 1.
    quadrants = {
        (0, 0): label_data[0:4],   # Top-left
        (0, 32): label_data[4:8],  # Top-right
        (32, 0): label_data[8:12], # Bottom-left
        (32, 32): label_data[12:16] # Bottom-right
    }

    # Draw the main 64x64 border
    draw_rect(-0.5, -0.5, 64)

    # If any label is > 1, it means the 64x64 is split into 32x32s
    if np.any(label_data > 1):
        draw_rect(31.5, -0.5, 32)  # Draw right 32x32 block
        draw_rect(-0.5, 31.5, 32)  # Draw bottom 32x32 block
    
    # Iterate through the sixteen 16x16 CU positions
    for i in range(16):
        # Calculate the top-left corner of the 16x16 block
        row = (i // 4) * 16
        col = (i % 4) * 16
        
        # If the label is 2, it's a 16x16 block. Draw its border if not already part of a 32x32.
        if label_data[i] >= 2:
            draw_rect(col - 0.5, row - 0.5, 16)

        # If the label is 3, it's split into 8x8 blocks. Draw the inner cross.
        if label_data[i] == 3:
            draw_rect(col - 0.5, row - 0.5, 8)       # Top-left 8x8
            draw_rect(col + 7.5, row - 0.5, 8)      # Top-right 8x8
            draw_rect(col - 0.5, row + 7.5, 8)      # Bottom-left 8x8
            draw_rect(col + 7.5, row + 7.5, 8)      # Bottom-right 8x8


def visualize_partitions_on_sample(file_path, sample_index, qp):
    """
    Reads a sample, extracts the image and label, and saves a visualization
    of the CU partitions overlaid on the image.
    """
    try:
        with open(file_path, 'rb') as f:
            start_byte = sample_index * NUM_SAMPLE_LENGTH
            f.seek(start_byte)
            sample_bytes = f.read(NUM_SAMPLE_LENGTH)
            
            if len(sample_bytes) < NUM_SAMPLE_LENGTH:
                print(f"Error: Could not read a full sample at index {sample_index}.")
                return

            data_array = np.frombuffer(sample_bytes, dtype=np.uint8)
            image_matrix = data_array[0:IMAGE_BYTES].reshape(IMAGE_SIZE, IMAGE_SIZE)
            
            labels_start_index = IMAGE_BYTES + 64
            label_offset = qp * NUM_LABEL_BYTES
            label_start = labels_start_index + label_offset
            label_data = data_array[label_start : label_start + NUM_LABEL_BYTES]

            # --- Create and save the visualization ---
            fig, ax = plt.subplots(1)
            ax.imshow(image_matrix, cmap='gray')
            
            # Draw the partitions on the image
            draw_cu_partitions(ax, label_data)

            ax.set_title(f"CU Partitions for Sample #{sample_index}, QP={qp}")
            ax.axis('off') # Hide the axes ticks

            output_filename = f'partition_sample_{sample_index}_qp_{qp}.png'
            plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1)
            plt.close()

            print(f"--- Visualization Complete ---")
            print(f"Label: {label_data}")
            print(f"✅ Partition image saved as '{output_filename}'")

    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Run the visualization ---
if __name__ == "__main__":
    visualize_partitions_on_sample(FILE_PATH, SAMPLE_INDEX, QP_TO_INSPECT)