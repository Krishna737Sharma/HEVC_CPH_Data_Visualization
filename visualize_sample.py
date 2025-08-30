import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---

# Path to your dataset file
FILE_PATH = '/workspaces/HEVC_CPH_Data_Visualization/AI_Valid_143925.dat'

# The index of the sample you want to see (0 for the first, 1 for the second, etc.)
SAMPLE_INDEX = 50 

# The Quantization Parameter (QP) you want to inspect the label for (0-51)
# The training code often uses 22, 27, 32, or 37.
QP_TO_INSPECT = 32

# --- Data Structure Constants (derived from your Python files) ---
IMAGE_SIZE = 64
NUM_CHANNELS = 1
NUM_LABEL_BYTES = 16

# Total size of the image data in bytes (64 * 64 * 1)
IMAGE_BYTES = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS

# Total size of one complete sample in the file
# 4096 (image) + 64 (unused) + (52 QPs * 16 bytes/label) = 4992
NUM_SAMPLE_LENGTH = 4992

def visualize_data_sample(file_path, sample_index, qp):
    """
    Reads a single sample from the .dat file, saves its image,
    and prints the label for a specific QP.
    """
    if not (0 <= qp <= 51):
        print("Error: QP_TO_INSPECT must be between 0 and 51.")
        return

    try:
        with open(file_path, 'rb') as f:
            # Calculate the starting position of the desired sample
            start_byte = sample_index * NUM_SAMPLE_LENGTH
            f.seek(start_byte)
            
            # Read the entire 4992-byte chunk for this one sample
            sample_bytes = f.read(NUM_SAMPLE_LENGTH)
            
            if len(sample_bytes) < NUM_SAMPLE_LENGTH:
                print(f"Error: Could not read a full sample at index {sample_index}. File might be too short.")
                return

            # Convert the byte string into a NumPy array of unsigned 8-bit integers
            data_array = np.frombuffer(sample_bytes, dtype=np.uint8)

            # --- 1. Extract and save the image ---
            image_data = data_array[0:IMAGE_BYTES]
            image_matrix = image_data.reshape(IMAGE_SIZE, IMAGE_SIZE)
            
            print(f"--- Processing Image for Sample #{sample_index} ---")
            plt.imshow(image_matrix, cmap='gray')
            plt.title(f"Image for Sample #{sample_index}")
            
            # This is the new part: saving the image to a file
            output_filename = f'sample_{sample_index}_qp_{qp}.png'
            plt.savefig(output_filename)
            print(f"Image saved as '{output_filename}'")
            plt.close() # Prevents the image from displaying in some environments

            # --- 2. Extract and print the label for the specified QP ---
            # The label block starts after the image (4096 bytes) and the unused block (64 bytes)
            labels_start_index = IMAGE_BYTES + 64 
            
            # Find the start of the specific QP's label within the label block
            label_offset = qp * NUM_LABEL_BYTES
            label_start = labels_start_index + label_offset
            label_end = label_start + NUM_LABEL_BYTES
            
            label_data = data_array[label_start:label_end]
            
            print(f"\n--- Label Data for QP = {qp} ---")
            print(label_data)
            print(f"\nThis 16-byte array represents the CU split decisions for this 64x64 block at QP {qp}.")

    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Run the visualization ---
if __name__ == "__main__":
    visualize_data_sample(FILE_PATH, SAMPLE_INDEX, QP_TO_INSPECT)