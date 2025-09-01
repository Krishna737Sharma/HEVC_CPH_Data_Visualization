import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Configuration ---

# Path to your dataset file
FILE_PATH = '/workspaces/HEVC_CPH_Data_Visualization/AI_Valid_143925.dat'

# The index of the sample you want to see (e.g., 0, 50, 100)
SAMPLE_INDEX = 51

# The Quantization Parameter (QP) you want to inspect (0-51)
QP_TO_INSPECT = 22

# --- Data Structure Constants ---
IMAGE_SIZE = 64
NUM_SAMPLE_LENGTH = 4992
IMAGE_BYTES = 4096
NUM_LABEL_BYTES = 16

def draw_cu_partitions(ax, label_data):
    """
    Corrected function to draw CU partitions for one 64x64 block.
    """
    label_grid = label_data.reshape(4, 4)
    
    def draw_rect(x, y, size, color='yellow', lw=1.2):
        # Helper to draw a rectangle with its top-left corner at (x,y)
        rect = patches.Rectangle((x - 0.5, y - 0.5), size, size, linewidth=lw, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    # If the whole CTU is one block (all labels are 0)
    if np.all(label_grid == 0):
        draw_rect(0, 0, 64)
        return

    # Iterate through the four 32x32 quadrants of the CTU
    for q_r in range(2): # Quadrant row (0 or 1)
        for q_c in range(2): # Quadrant column (0 or 1)
            # Get the 4 labels for this quadrant
            sub_grid = label_grid[q_r*2:(q_r+1)*2, q_c*2:(q_c+1)*2]
            
            # If this quadrant is a single 32x32 block (all labels are 1)
            if np.all(sub_grid == 1):
                draw_rect(q_c*32, q_r*32, 32)
            else: # If the quadrant is split further, check its 16x16 blocks
                for r in range(2):
                    for c in range(2):
                        label = sub_grid[r, c]
                        x_16 = q_c*32 + c*16
                        y_16 = q_r*32 + r*16
                        
                        if label == 2: # This is a 16x16 block
                            draw_rect(x_16, y_16, 16)
                        elif label == 3: # This 16x16 is split into 8x8
                            draw_rect(x_16, y_16, 8)
                            draw_rect(x_16 + 8, y_16, 8)
                            draw_rect(x_16, y_16 + 8, 8)
                            draw_rect(x_16 + 8, y_16 + 8, 8)


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