import numpy as np
import pandas as pd
import os

# --- Configuration ---
FILE_PATH = '/workspaces/HEVC_CPH_Data_Visualization/AI_Valid_143925.dat'
SAMPLES_TO_PROCESS = 10000  # Process the first 10,000 samples
QP_TO_INSPECT = 22 # Which QP's labels to include in the summary
OUTPUT_CSV_FILE = 'data_summary.csv'

# --- Data Structure Constants ---
NUM_SAMPLE_LENGTH = 4992
IMAGE_BYTES = 4096
NUM_LABEL_BYTES = 16

def create_dataset_summary(file_path, num_samples, qp):
    """
    Reads a subset of the .dat file and saves a statistical summary
    and sample data to a CSV file.
    """
    print(f"Analyzing the first {num_samples} samples for QP={qp}...")
    
    # --- Step 1: Calculate total samples in the file ---
    try:
        file_size = os.path.getsize(file_path)
        total_samples = file_size // NUM_SAMPLE_LENGTH
        print(f"Total samples found in file: {total_samples:,}")
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Lists to hold the extracted summary data
    avg_pixel_values = []
    labels_data = []

    # --- Step 2: Read and process the subset of samples ---
    with open(file_path, 'rb') as f:
        for i in range(min(num_samples, total_samples)):
            sample_bytes = f.read(NUM_SAMPLE_LENGTH)
            if not sample_bytes:
                break # Stop if we reach the end of the file

            data_array = np.frombuffer(sample_bytes, dtype=np.uint8)

            # Calculate average pixel intensity for the image
            image_data = data_array[0:IMAGE_BYTES]
            avg_pixel_values.append(image_data.mean())

            # Extract the 16-byte label for the specified QP
            labels_start_index = IMAGE_BYTES + 64
            label_offset = qp * NUM_LABEL_BYTES
            label_start = labels_start_index + label_offset
            label_end = label_start + NUM_LABEL_BYTES
            labels_data.append(data_array[label_start:label_end])

    # --- Step 3: Create a Pandas DataFrame and save to CSV ---
    if not avg_pixel_values:
        print("No data was processed.")
        return
        
    # Create a DataFrame from the collected data
    df = pd.DataFrame({
        'avg_pixel_intensity': avg_pixel_values
    })
    
    # Create column names for the 16 label bytes
    label_columns = {f'label_byte_{i}': [label[i] for label in labels_data] for i in range(NUM_LABEL_BYTES)}
    labels_df = pd.DataFrame(label_columns)
    
    # Combine the two dataframes
    summary_df = pd.concat([df, labels_df], axis=1)

    # Display statistical summary in the console
    print("\n--- Statistical Summary of Processed Samples ---")
    print(summary_df.describe())

    # Save the summary DataFrame to a CSV file
    summary_df.to_csv(OUTPUT_CSV_FILE, index_label='sample_index')
    print(f"\n✅ Summary of {len(summary_df)} samples saved to '{OUTPUT_CSV_FILE}'")

# --- Run the summary creation ---
if __name__ == "__main__":
    create_dataset_summary(FILE_PATH, SAMPLES_TO_PROCESS, QP_TO_INSPECT)