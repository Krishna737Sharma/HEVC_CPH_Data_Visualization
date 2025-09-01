import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import data_info as di # Assumes data_info.py is in the same folder

# --- 1. Configuration: SET THESE VALUES ---

# Paths to your data folders
YUV_PATH = '/workspaces/HEVC_CPH_Data_Visualization/yuv'
INFO_PATH = '/workspaces/HEVC_CPH_Data_Visualization/info'

# Details of the frame to inspect
VIDEO_FILENAME = 'IntraTrain_768x512'
FRAME_TO_VISUALIZE = 3
QP_TO_INSPECT = 22

# The specific pixel coordinate you want to inspect
X_COORDINATE = 112
Y_COORDINATE = 160

# --- 2. Helper Functions (from previous scripts) ---

class FrameYUV:
    def __init__(self, Y, U, V):
        self._Y = Y

def read_YUV420_frame(fid, width, height, frame_index):
    frame_bytes = (width * height * 3) // 2
    fid.seek(frame_index * frame_bytes)
    Y_buf = fid.read(width * height)
    if not Y_buf: return None
    Y = np.frombuffer(Y_buf, dtype=np.uint8).reshape([height, width])
    return FrameYUV(Y, None, None)

def read_info_frame(fid, width, height, frame_index):
    num_units_wide = width // 16
    num_units_high = height // 16
    frame_bytes = num_units_wide * num_units_high
    fid.seek(frame_index * frame_bytes)
    info_buf = fid.read(frame_bytes)
    if not info_buf: return None
    info = np.frombuffer(info_buf, dtype=np.uint8).reshape([num_units_high, num_units_wide])
    return info

def draw_cu_partitions(ax, label_data, x_offset, y_offset):
    label_grid = label_data.reshape(4, 4)
    def draw_rect(x, y, size, color='yellow', lw=1.2):
        rect = patches.Rectangle((x - 0.5, y - 0.5), size, size, linewidth=lw, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    if np.all(label_grid == 0):
        draw_rect(x_offset, y_offset, 64)
        return

    for q_r in range(2):
        for q_c in range(2):
            sub_grid = label_grid[q_r*2:(q_r+1)*2, q_c*2:(q_c+1)*2]
            if np.all(sub_grid == 1):
                draw_rect(x_offset + q_c*32, y_offset + q_r*32, 32)
            else:
                for r in range(2):
                    for c in range(2):
                        label = sub_grid[r, c]
                        x_16 = x_offset + q_c*32 + c*16
                        y_16 = y_offset + q_r*32 + r*16
                        if label == 2:
                            draw_rect(x_16, y_16, 16)
                        elif label == 3:
                            draw_rect(x_16, y_16, 8)
                            draw_rect(x_16 + 8, y_16, 8)
                            draw_rect(x_16, y_16 + 8, 8)
                            draw_rect(x_16 + 8, y_16 + 8, 8)

# --- 3. Main Inspection Logic ---

def inspect_coordinate_partitions():
    try:
        video_index = di.YUV_NAME_LIST_FULL.index(VIDEO_FILENAME)
        width = di.YUV_WIDTH_LIST_FULL[video_index]
        height = di.YUV_HEIGHT_LIST_FULL[video_index]
    except ValueError:
        print(f"Error: Video '{VIDEO_FILENAME}' not found in data_info.py.")
        return

    # Input validation
    if not (0 <= X_COORDINATE < width and 0 <= Y_COORDINATE < height):
        print(f"Error: Coordinate ({X_COORDINATE}, {Y_COORDINATE}) is outside the frame dimensions ({width}x{height}).")
        return

    # Find file paths
    yuv_file_path = os.path.join(YUV_PATH, VIDEO_FILENAME + '.yuv')
    import glob
    info_file_pattern = os.path.join(INFO_PATH, f'Info*_{VIDEO_FILENAME}_*qp{QP_TO_INSPECT}*CUDepth.dat')
    info_files = glob.glob(info_file_pattern)
    if not info_files:
        print(f"Error: Could not find Info file for QP {QP_TO_INSPECT}.")
        return
    info_file_path = info_files[0]
    
    try:
        # Read the full frame data once
        with open(yuv_file_path, 'rb') as fid_yuv:
            frame_yuv = read_YUV420_frame(fid_yuv, width, height, FRAME_TO_VISUALIZE)
        with open(info_file_path, 'rb') as fid_info:
            cu_depth_map = read_info_frame(fid_info, width, height, FRAME_TO_VISUALIZE)

        if frame_yuv is None or cu_depth_map is None:
            print(f"Error: Could not read frame #{FRAME_TO_VISUALIZE}.")
            return

        # --- Locate and Extract the Specific Block ---
        ctu_x_index = X_COORDINATE // 64
        ctu_y_index = Y_COORDINATE // 64

        # Extract 64x64 image patch
        x_start, y_start = ctu_x_index * 64, ctu_y_index * 64
        image_patch = frame_yuv._Y[y_start:y_start+64, x_start:x_start+64]

        # Extract 4x4 partition label
        map_x_start, map_y_start = ctu_x_index * 4, ctu_y_index * 4
        partition_label = cu_depth_map[map_y_start:map_y_start+4, map_x_start:map_x_start+4]
        
        print(f"Coordinate ({X_COORDINATE}, {Y_COORDINATE}) is in the 64x64 block at grid position ({ctu_x_index}, {ctu_y_index}).")

        # --- Save the Text File ---
        txt_filename = f'partition_label_X{X_COORDINATE}_Y{Y_COORDINATE}_qp{QP_TO_INSPECT}.txt'
        np.savetxt(txt_filename, partition_label, fmt='%d', delimiter=' ')
        print(f"✅ Partition label map saved to '{txt_filename}'")

        # --- Save the Visualization PNG ---
        png_filename = f'partition_viz_X{X_COORDINATE}_Y{Y_COORDINATE}_qp{QP_TO_INSPECT}.png'
        fig, ax = plt.subplots(1, figsize=(6, 6))
        ax.imshow(image_patch, cmap='gray')
        draw_cu_partitions(ax, partition_label.flatten(), x_offset=0, y_offset=0) # Draw with no offset
        ax.set_title(f"Partitions at ({X_COORDINATE}, {Y_COORDINATE})")
        ax.axis('off')
        plt.savefig(png_filename, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"✅ Partition visualization saved to '{png_filename}'")


    except FileNotFoundError as e:
        print(f"Error: File not found. Please check your paths.\n{e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    inspect_coordinate_partitions()