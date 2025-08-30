import numpy as np
import os
import data_info as di # Assumes data_info.py is in the same folder

# --- 1. Configuration: SET THESE VALUES ---

# Path to the folder containing your Info_XX.dat files
INFO_PATH = '/workspaces/HEVC_CPH_Data_Visualization/info' # Or your correct path

# Name of the video/image file to get the map for
VIDEO_FILENAME = 'IntraTrain_768x512'

# Frame/image number to extract from the file (0 is the first)
FRAME_TO_EXTRACT = 3

# QP for the partition map you want to save (22, 27, 32, or 37)
QP_TO_INSPECT = 22

# --- 2. Helper Function (from your scripts) ---

def read_info_frame(fid, width, height, frame_index):
    """Reads the CU depth map for a specific frame."""
    num_units_wide = width // 16
    num_units_high = height // 16
    frame_bytes = num_units_wide * num_units_high
    
    # Seek to the start of the desired frame's data
    fid.seek(frame_index * frame_bytes)
    
    info_buf = fid.read(frame_bytes)
    if not info_buf: return None # End of file
    
    info = np.frombuffer(info_buf, dtype=np.uint8).reshape([num_units_high, num_units_wide])
    return info

# --- 3. Main Logic to Extract and Save the Map ---

def save_partition_map_to_text():
    try:
        # Find video properties from data_info.py
        try:
            video_index = di.YUV_NAME_LIST_FULL.index(VIDEO_FILENAME)
            width = di.YUV_WIDTH_LIST_FULL[video_index]
            height = di.YUV_HEIGHT_LIST_FULL[video_index]
        except ValueError:
            print(f"Error: Video '{VIDEO_FILENAME}' not found in data_info.py.")
            return

        # Find the specific info file
        import glob
        info_file_pattern = os.path.join(INFO_PATH, f'Info*_{VIDEO_FILENAME}_*qp{QP_TO_INSPECT}*CUDepth.dat')
        info_files = glob.glob(info_file_pattern)
        if not info_files:
            print(f"Error: Could not find Info file for QP {QP_TO_INSPECT}.")
            return
        info_file_path = info_files[0]
        
        print(f"Reading partition map for frame #{FRAME_TO_EXTRACT} from '{info_file_path}'")

        with open(info_file_path, 'rb') as fid_info:
            cu_depth_map = read_info_frame(fid_info, width, height, FRAME_TO_EXTRACT)
        
        if cu_depth_map is not None:
            # Define the output filename
            output_filename = f'partition_map_{VIDEO_FILENAME}_frame_{FRAME_TO_EXTRACT}_qp_{QP_TO_INSPECT}.txt'
            
            # Save the 2D numpy array to a text file
            np.savetxt(output_filename, cu_depth_map, fmt='%d', delimiter=' ')
            
            print(f"\n✅ Partition map successfully saved to '{output_filename}'")
            print(f"   The text file has a shape of {cu_depth_map.shape} (Height x Width in 16x16 blocks).")

        else:
            print(f"Error: Frame #{FRAME_TO_EXTRACT} could not be read. File may be too short.")

    except FileNotFoundError:
        print(f"Error: A file was not found. Please check that your INFO_PATH is correct and the files exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    save_partition_map_to_text()