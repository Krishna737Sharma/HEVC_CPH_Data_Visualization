import numpy as np
import os
import data_info as di # Assumes data_info.py is in the same folder

# --- 1. Configuration: Choose your frame ---
YUV_PATH = '/workspaces/HEVC_CPH_Data_Visualization/yuv'         # Your path to YUV files
INFO_PATH = '/workspaces/HEVC_CPH_Data_Visualization/info' # Your path to Info files

VIDEO_FILENAME = 'IntraTrain_768x512'
FRAME_TO_EXTRACT = 7  # 0 for the first frame/image
QP_TO_INSPECT = 22

# --- 2. Core Functions (copied from your extract_data_AI.py) ---

class FrameYUV:
    def __init__(self, Y, U, V):
        self._Y = Y

def read_YUV420_frame(fid, width, height):
    # Reads one frame sequentially
    d00 = height // 2
    d01 = width // 2
    Y_buf = fid.read(width * height)
    if not Y_buf: return None # End of file
    Y = np.frombuffer(Y_buf, dtype=np.uint8).reshape([height, width])
    # Skip U and V for this example
    fid.read(d01 * d00) # U
    fid.read(d01 * d00) # V
    return FrameYUV(Y, None, None)

def read_info_frame(fid, width, height, mode):
    # Reads one frame's info sequentially
    if mode == 'CU': unit_width = 16
    else: unit_width = 8
    
    num_line_in_unit = height // unit_width
    num_column_in_unit = width // unit_width
    info_buf = fid.read(num_line_in_unit * num_column_in_unit)
    if not info_buf: return None # End of file
    info = np.frombuffer(info_buf, dtype=np.uint8).reshape([num_line_in_unit, num_column_in_unit])
    return info

# --- 3. Main Extraction and Printing Logic ---

# Find video properties
try:
    video_index = di.YUV_NAME_LIST_FULL.index(VIDEO_FILENAME)
    width = di.YUV_WIDTH_LIST_FULL[video_index]
    height = di.YUV_HEIGHT_LIST_FULL[video_index]
except ValueError:
    print(f"Error: Video '{VIDEO_FILENAME}' not found in data_info.py.")
    exit()

# Find the specific info file
import glob
info_file_pattern = os.path.join(INFO_PATH, f'Info*_{VIDEO_FILENAME}_*qp{QP_TO_INSPECT}*CUDepth.dat')
info_files = glob.glob(info_file_pattern)
if not info_files:
    print(f"Error: Could not find Info file for QP {QP_TO_INSPECT}.")
    exit()
info_file_path = info_files[0]
yuv_file_path = os.path.join(YUV_PATH, VIDEO_FILENAME + '.yuv')

try:
    with open(yuv_file_path, 'rb') as fid_yuv, open(info_file_path, 'rb') as fid_info:
        # Loop to get to the desired frame
        for i in range(FRAME_TO_EXTRACT + 1):
            frame_yuv = read_YUV420_frame(fid_yuv, width, height)
            cu_depth_map = read_info_frame(fid_info, width, height, 'CU')
        
        if frame_yuv and cu_depth_map is not None:
            print(f"--- Extraction successful for Frame #{FRAME_TO_EXTRACT} ---")
            
            # Print Frame Pixel Data Info
            frame_pixels = frame_yuv._Y
            print(f"\n1. Frame Pixel Data:")
            print(f"   - Shape: {frame_pixels.shape}")
            print(f"   - Data Type: {frame_pixels.dtype}")
            
            # Print Partition Label Info
            print(f"\n2. Partition Labels (CU Depth Map) for QP={QP_TO_INSPECT}:")
            print(f"   - Shape: {cu_depth_map.shape}")
            print(f"   - Data Type: {cu_depth_map.dtype}")
            
            print("\n   - Snippet of Partition Labels (top-left 4x4 corner):")
            print(cu_depth_map[0:4, 0:4])

        else:
            print(f"Error: Frame #{FRAME_TO_EXTRACT} could not be read. File may be too short.")

except FileNotFoundError:
    print(f"Error: Make sure files exist at '{yuv_file_path}' and '{info_file_path}'")