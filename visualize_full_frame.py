import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import data_info as di # Uses the data_info.py file you provided

# --- 1. Configuration: SET THESE VALUES ---

# Path to the folder containing your original .yuv files
YUV_PATH = '/workspaces/HEVC_CPH_Data_Visualization/yuv'

# Path to the folder containing your Info_XX.dat files
INFO_PATH = '/workspaces/HEVC_CPH_Data_Visualization/info'

# Name of the video file you want to visualize (WITHOUT .yuv)
VIDEO_FILENAME = 'IntraTrain_768x512'

# Frame number to extract from the video (0 is the first frame)
FRAME_TO_VISUALIZE = 3

# QP to inspect the partitions for (must be one of [22, 27, 32, 37])
QP_TO_INSPECT = 22

# --- 2. Helper Functions ---

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
    """
    Corrected function to draw CU partitions for one 64x64 block.
    """
    label_grid = label_data.reshape(4, 4)
    
    def draw_rect(x, y, size, color='yellow', lw=0.8):
        # Helper to draw a rectangle with its center at (x,y)
        rect = patches.Rectangle((x - 0.5, y - 0.5), size, size, linewidth=lw, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    # If the whole CTU is one block (all labels are 0)
    if np.all(label_grid == 0):
        draw_rect(x_offset, y_offset, 64, lw=1.5)
        return

    # Iterate through the four 32x32 quadrants of the CTU
    for q_r in range(2): # Quadrant row (0 or 1)
        for q_c in range(2): # Quadrant column (0 or 1)
            # Get the 4 labels for this quadrant
            sub_grid = label_grid[q_r*2:(q_r+1)*2, q_c*2:(q_c+1)*2]
            
            # If this quadrant is a single 32x32 block (all labels are 1)
            if np.all(sub_grid == 1):
                draw_rect(x_offset + q_c*32, y_offset + q_r*32, 32, lw=1.2)
            else: # If the quadrant is split further, check its 16x16 blocks
                for r in range(2):
                    for c in range(2):
                        label = sub_grid[r, c]
                        x_16 = x_offset + q_c*32 + c*16
                        y_16 = y_offset + q_r*32 + r*16
                        
                        if label == 2: # This is a 16x16 block
                            draw_rect(x_16, y_16, 16)
                        elif label == 3: # This 16x16 is split into 8x8
                            draw_rect(x_16, y_16, 8)
                            draw_rect(x_16 + 8, y_16, 8)
                            draw_rect(x_16, y_16 + 8, 8)
                            draw_rect(x_16 + 8, y_16 + 8, 8)


# --- 3. Main Visualization Logic ---

def generate_full_frame_visualization():
    try:
        video_index = di.YUV_NAME_LIST_FULL.index(VIDEO_FILENAME)
        width = di.YUV_WIDTH_LIST_FULL[video_index]
        height = di.YUV_HEIGHT_LIST_FULL[video_index]
    except ValueError:
        print(f"Error: Video '{VIDEO_FILENAME}' not found in data_info.py.")
        return

    yuv_file_path = os.path.join(YUV_PATH, VIDEO_FILENAME + '.yuv')
    import glob
    info_file_pattern = os.path.join(INFO_PATH, f'Info*_{VIDEO_FILENAME}_*qp{QP_TO_INSPECT}*CUDepth.dat')
    info_files = glob.glob(info_file_pattern)
    if not info_files:
        print(f"Error: Could not find Info file for QP {QP_TO_INSPECT}. Pattern: {info_file_pattern}")
        return
    info_file_path = info_files[0]
    
    print(f"Reading frame {FRAME_TO_VISUALIZE} from '{yuv_file_path}'")
    print(f"Reading partitions from '{info_file_path}'")

    try:
        with open(yuv_file_path, 'rb') as fid_yuv:
            frame_yuv = read_YUV420_frame(fid_yuv, width, height, FRAME_TO_VISUALIZE)
        with open(info_file_path, 'rb') as fid_info:
            cu_depth_map = read_info_frame(fid_info, width, height, FRAME_TO_VISUALIZE)

        if frame_yuv is None or cu_depth_map is None:
            print(f"Error: Could not read frame #{FRAME_TO_VISUALIZE}. File may not contain that many frames.")
            return

        fig, ax = plt.subplots(1, figsize=(width / 100, height / 100))
        ax.imshow(frame_yuv._Y, cmap='gray')
        
        for y_ctu in range(height // 64):
            for x_ctu in range(width // 64):
                ctu_label = cu_depth_map[y_ctu*4:(y_ctu+1)*4, x_ctu*4:(x_ctu+1)*4].flatten()
                draw_cu_partitions(ax, ctu_label, x_offset=x_ctu*64, y_offset=y_ctu*64)

        ax.axis('off')
        output_filename = f'frame_{FRAME_TO_VISUALIZE}_{VIDEO_FILENAME}_qp_{QP_TO_INSPECT}.png'
        plt.savefig(output_filename, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"\n✅ Full frame visualization saved as '{output_filename}'")

    except FileNotFoundError as e:
        print(f"Error: File not found. Please check your paths.\n{e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    generate_full_frame_visualization()