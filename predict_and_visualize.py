import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import data_info as di
from Vit_model import VisionTransformer # Imports the model definition from your file

# --- 1. Configuration: SET THESE VALUES ---

# Paths to your data and model
YUV_PATH = '/workspaces/HEVC_CPH_Data_Visualization/yuv'
INFO_PATH = '/workspaces/HEVC_CPH_Data_Visualization/info'
MODEL_CHECKPOINT_PATH = '/workspaces/HEVC_CPH_Data_Visualization/best_vit_model.pth' # Path to your trained model checkpoint

# Details of the frame to inspect
VIDEO_FILENAME = 'IntraTrain_768x512'
FRAME_TO_VISUALIZE = 3
QP_TO_INSPECT = 22 # The QP must be one of [22, 27, 32, 37]

# The specific pixel coordinate you want to inspect
X_COORDINATE = 112
Y_COORDINATE = 160

# --- 2. Helper Functions ---

def read_YUV420_frame(fid, width, height, frame_index):
    frame_bytes = (width * height * 3) // 2
    fid.seek(frame_index * frame_bytes)
    Y_buf = fid.read(width * height)
    if not Y_buf: return None
    Y = np.frombuffer(Y_buf, dtype=np.uint8).reshape([height, width])
    return Y

def read_info_frame(fid, width, height, frame_index):
    num_units_wide = width // 16
    num_units_high = height // 16
    frame_bytes = num_units_wide * num_units_high
    fid.seek(frame_index * frame_bytes)
    info_buf = fid.read(frame_bytes)
    if not info_buf: return None
    return np.frombuffer(info_buf, dtype=np.uint8).reshape([num_units_high, num_units_wide])

def draw_cu_partitions(ax, label_data, x_offset, y_offset):
    # This function is the same as in the previous script
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
                            draw_rect(x_16, y_16, 8); draw_rect(x_16 + 8, y_16, 8)
                            draw_rect(x_16, y_16 + 8, 8); draw_rect(x_16 + 8, y_16 + 8, 8)

def decode_prediction_to_depth_map(prediction_vector):
    """
    Converts the model's 21-element prediction vector back into a 16-element
    CU depth map (0, 1, 2, 3) for visualization.
    """
    pred = (prediction_vector > 0.5).float().cpu().numpy().flatten()
    depth_map = np.zeros(16, dtype=int)

    # If 64x64 is split
    if pred[0] == 1:
        depth_map[:] = 1
        
        # Check 32x32 splits
        for i in range(4): # For each of the 4 quadrants
            if pred[1 + i] == 1:
                start_idx = (i//2)*8 + (i%2)*2 # 0, 2, 8, 10
                depth_map[start_idx:start_idx+2] = 2
                depth_map[start_idx+4:start_idx+6] = 2

    # Check 16x16 splits
    for i in range(16):
        if pred[5 + i] == 1:
            depth_map[i] = 3
            
    return depth_map

# --- 3. Main Prediction and Visualization Logic ---

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    print(f"Loading model from '{MODEL_CHECKPOINT_PATH}'...")
    model = VisionTransformer(embed_dim=196, depth=5, num_heads=4).to(device) # Ensure params match your trained model
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully.")

    # --- Get Data ---
    try:
        video_index = di.YUV_NAME_LIST_FULL.index(VIDEO_FILENAME)
        width, height = di.YUV_WIDTH_LIST_FULL[video_index], di.YUV_HEIGHT_LIST_FULL[video_index]
        yuv_file = os.path.join(YUV_PATH, VIDEO_FILENAME + '.yuv')
        info_file = glob.glob(os.path.join(INFO_PATH, f'Info*_{VIDEO_FILENAME}_*qp{QP_TO_INSPECT}*CUDepth.dat'))[0]
        
        with open(yuv_file, 'rb') as f_yuv:
            frame_pixels = read_YUV420_frame(f_yuv, width, height, FRAME_TO_VISUALIZE)
        with open(info_file, 'rb') as f_info:
            full_depth_map = read_info_frame(f_info, width, height, FRAME_TO_VISUALIZE)
    except (ValueError, FileNotFoundError, IndexError) as e:
        print(f"Error loading data: {e}")
        return

    # --- Extract Block ---
    ctu_x_idx, ctu_y_idx = X_COORDINATE // 64, Y_COORDINATE // 64
    x_start, y_start = ctu_x_idx * 64, ctu_y_idx * 64
    image_patch = frame_pixels[y_start:y_start+64, x_start:x_start+64]
    
    map_x_start, map_y_start = ctu_x_idx * 4, ctu_y_idx * 4
    ground_truth_label = full_depth_map[map_y_start:map_y_start+4, map_x_start:map_x_start+4].flatten()

    # --- Prepare for Prediction ---
    image_tensor = torch.tensor(image_patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    qp_tensor = torch.tensor([float(QP_TO_INSPECT) / 51.0], dtype=torch.float32)
    image_tensor, qp_tensor = image_tensor.to(device), qp_tensor.to(device)

    # --- Run Prediction ---
    with torch.no_grad():
        prediction_vector = model(image_tensor, qp_tensor)

    # --- Decode and Save ---
    predicted_label = decode_prediction_to_depth_map(prediction_vector)
    
    print(f"\nCoordinate ({X_COORDINATE}, {Y_COORDINATE}) in CTU ({ctu_x_idx}, {ctu_y_idx})")
    print(f"Ground Truth Label: {ground_truth_label}")
    print(f"Predicted Label:    {predicted_label}")
    
    # Save text files
    np.savetxt(f'pred_ground_truth_X{X_COORDINATE}_Y{Y_COORDINATE}.txt', ground_truth_label.reshape(4,4), fmt='%d')
    np.savetxt(f'pred_model_output_X{X_COORDINATE}_Y{Y_COORDINATE}.txt', predicted_label.reshape(4,4), fmt='%d')

    # --- Create Side-by-Side Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Ground Truth Plot
    ax1.imshow(image_patch, cmap='gray')
    draw_cu_partitions(ax1, ground_truth_label, x_offset=0, y_offset=0)
    ax1.set_title("Ground Truth Partitions")
    ax1.axis('off')

    # Model Prediction Plot
    ax2.imshow(image_patch, cmap='gray')
    draw_cu_partitions(ax2, predicted_label, x_offset=0, y_offset=0)
    ax2.set_title("Model Prediction")
    ax2.axis('off')
    
    output_filename = f'pred_comparison_X{X_COORDINATE}_Y{Y_COORDINATE}_qp{QP_TO_INSPECT}.png'
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"\n✅ Comparison visualization saved as '{output_filename}'")

if __name__ == "__main__":
    main()