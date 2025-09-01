import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import data_info as di
from Vit_model import VisionTransformer # Imports the model definition

# --- 1. Configuration: SET THESE VALUES ---

# Paths to your data and model
YUV_PATH = '/workspaces/HEVC_CPH_Data_Visualization/yuv'
INFO_PATH = '/workspaces/HEVC_CPH_Data_Visualization/info'
MODEL_CHECKPOINT_PATH = 'best_vit_model.pth' # Path to your trained model checkpoint

# Details of the frame to predict
VIDEO_FILENAME = 'IntraTrain_768x512'
FRAME_TO_VISUALIZE = 3
QP_TO_INSPECT = 22 # The QP must be one of [22, 27, 32, 37]

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
    label_grid = label_data.reshape(4, 4)
    def draw_rect(x, y, size, color='yellow', lw=0.8):
        rect = patches.Rectangle((x - 0.5, y - 0.5), size, size, linewidth=lw, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
    if np.all(label_grid == 0):
        draw_rect(x_offset, y_offset, 64, lw=1.5); return
    for q_r in range(2):
        for q_c in range(2):
            sub_grid = label_grid[q_r*2:(q_r+1)*2, q_c*2:(q_c+1)*2]
            if np.all(sub_grid == 1):
                draw_rect(x_offset + q_c*32, y_offset + q_r*32, 32, lw=1.2)
            else:
                for r in range(2):
                    for c in range(2):
                        label = sub_grid[r, c]
                        x_16, y_16 = x_offset + q_c*32 + c*16, y_offset + q_r*32 + r*16
                        if label == 2:
                            draw_rect(x_16, y_16, 16)
                        elif label == 3:
                            draw_rect(x_16, y_16, 8); draw_rect(x_16 + 8, y_16, 8)
                            draw_rect(x_16, y_16 + 8, 8); draw_rect(x_16 + 8, y_16 + 8, 8)

def decode_prediction_to_depth_map(prediction_vector):
    pred = (prediction_vector > 0.5).float().cpu().numpy().flatten()
    depth_map = np.zeros(16, dtype=int)
    if pred[0] == 1:
        depth_map[:] = 1
        for i in range(4):
            if pred[1 + i] == 1:
                start_idx = (i//2)*8 + (i%2)*2
                depth_map[start_idx:start_idx+2] = 2
                depth_map[start_idx+4:start_idx+6] = 2
    for i in range(16):
        if pred[5 + i] == 1:
            depth_map[i] = 3
    return depth_map

# --- 3. Main Prediction and Visualization Logic ---

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    print(f"Loading model from '{MODEL_CHECKPOINT_PATH}'...")
    model = VisionTransformer(embed_dim=196, depth=5, num_heads=4).to(device)
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
            ground_truth_map = read_info_frame(f_info, width, height, FRAME_TO_VISUALIZE)
    except (ValueError, FileNotFoundError, IndexError) as e:
        print(f"Error loading data: {e}"); return

    # --- Prepare for Loop ---
    full_predicted_map = np.zeros_like(ground_truth_map)
    qp_tensor = torch.tensor([float(QP_TO_INSPECT) / 51.0], dtype=torch.float32).to(device)
    
    print("\nProcessing frame... This may take a moment.")
    # --- Loop through frame and predict for each CTU ---
    for y_ctu in range(height // 64):
        for x_ctu in range(width // 64):
            # Extract 64x64 image patch
            x_start, y_start = x_ctu * 64, y_ctu * 64
            image_patch = frame_pixels[y_start:y_start+64, x_start:x_start+64]
            
            # Prepare tensor for model
            image_tensor = torch.tensor(image_patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
            image_tensor = image_tensor.to(device)
            
            # Run Prediction
            with torch.no_grad():
                prediction_vector = model(image_tensor, qp_tensor)
            
            # Decode and store the result
            predicted_label = decode_prediction_to_depth_map(prediction_vector)
            map_x_start, map_y_start = x_ctu * 4, y_ctu * 4
            full_predicted_map[map_y_start:map_y_start+4, map_x_start:map_x_start+4] = predicted_label.reshape(4,4)
    
    print("Frame processing complete.")

    # --- Save Text File of Prediction ---
    txt_filename = f'full_frame_prediction_qp{QP_TO_INSPECT}.txt'
    np.savetxt(txt_filename, full_predicted_map, fmt='%d')
    print(f"✅ Predicted partition map saved to '{txt_filename}'")

    # --- Create Side-by-Side Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width / 50, height / 50))
    
    # Ground Truth Plot
    ax1.imshow(frame_pixels, cmap='gray')
    for y_ctu in range(height // 64):
        for x_ctu in range(width // 64):
            label = ground_truth_map[y_ctu*4:(y_ctu+1)*4, x_ctu*4:(x_ctu+1)*4].flatten()
            draw_cu_partitions(ax1, label, x_offset=x_ctu*64, y_offset=y_ctu*64)
    ax1.set_title("Ground Truth Partitions")
    ax1.axis('off')

    # Model Prediction Plot
    ax2.imshow(frame_pixels, cmap='gray')
    for y_ctu in range(height // 64):
        for x_ctu in range(width // 64):
            label = full_predicted_map[y_ctu*4:(y_ctu+1)*4, x_ctu*4:(x_ctu+1)*4].flatten()
            draw_cu_partitions(ax2, label, x_offset=x_ctu*64, y_offset=y_ctu*64)
    ax2.set_title("Model Prediction")
    ax2.axis('off')
    
    output_filename = f'full_frame_comparison_qp{QP_TO_INSPECT}.png'
    fig.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close()
    
    print(f"✅ Comparison visualization saved as '{output_filename}'")

if __name__ == "__main__":
    main()