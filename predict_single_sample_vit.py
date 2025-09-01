import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from Vit_model import VisionTransformer # Imports the model definition from your file

# --- 1. Configuration: SET THESE VALUES ---

# Path to your .dat file
DAT_FILE_PATH = '/workspaces/HEVC_CPH_Data_Visualization/AI_Valid_143925.dat'

# Path to your trained ViT model checkpoint
MODEL_CHECKPOINT_PATH = 'best_vit_model.pth' 

# The index of the sample you want to predict (from your example script)
SAMPLE_INDEX = 51 

# The QP to use for prediction (must be one of [22, 27, 32, 37])
QP_TO_INSPECT = 22

# --- Data Structure Constants ---
IMAGE_SIZE = 64
IMAGE_BYTES = IMAGE_SIZE * IMAGE_SIZE
NUM_LABEL_BYTES = 16
NUM_SAMPLE_LENGTH = 4992

# --- 2. Helper Functions ---

def draw_cu_partitions(ax, label_data, x_offset, y_offset, color='yellow'):
    # This function is the same as in previous scripts
    label_grid = label_data.reshape(4, 4)
    def draw_rect(x, y, size, lw=1.2):
        rect = patches.Rectangle((x - 0.5, y - 0.5), size, size, linewidth=lw, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
    if np.all(label_grid == 0):
        draw_rect(x_offset, y_offset, 64); return
    for q_r in range(2):
        for q_c in range(2):
            sub_grid = label_grid[q_r*2:(q_r+1)*2, q_c*2:(q_c+1)*2]
            if np.all(sub_grid == 1):
                draw_rect(x_offset + q_c*32, y_offset + q_r*32, 32)
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
    """
    Converts the ViT model's 21-element prediction vector back into a
    16-element CU depth map (0, 1, 2, 3) for visualization.
    """
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
    try:
        print(f"Loading model from '{MODEL_CHECKPOINT_PATH}'...")
        model = VisionTransformer(embed_dim=196, depth=5, num_heads=4).to(device)
        checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at '{MODEL_CHECKPOINT_PATH}'. Please check the path.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    # --- Read the Single Sample ---
    try:
        with open(DAT_FILE_PATH, 'rb') as f:
            start_byte = SAMPLE_INDEX * NUM_SAMPLE_LENGTH
            f.seek(start_byte)
            sample_bytes = f.read(NUM_SAMPLE_LENGTH)
            if len(sample_bytes) < NUM_SAMPLE_LENGTH:
                print(f"Error: Could not read a full sample at index {SAMPLE_INDEX}.")
                return
        data_array = np.frombuffer(sample_bytes, dtype=np.uint8)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{DAT_FILE_PATH}'.")
        return
        
    # --- Extract Ground Truth and Image ---
    image_patch = data_array[0:IMAGE_BYTES].reshape(IMAGE_SIZE, IMAGE_SIZE)
    labels_start_index = IMAGE_BYTES + 64
    label_offset = QP_TO_INSPECT * NUM_LABEL_BYTES
    ground_truth_label = data_array[labels_start_index + label_offset : labels_start_index + label_offset + NUM_LABEL_BYTES]

    # --- Prepare for Prediction ---
    image_tensor = torch.tensor(image_patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    qp_tensor = torch.tensor([float(QP_TO_INSPECT) / 51.0], dtype=torch.float32)
    image_tensor, qp_tensor = image_tensor.to(device), qp_tensor.to(device)

    # --- Run Prediction ---
    with torch.no_grad():
        prediction_vector = model(image_tensor, qp_tensor)

    # --- Decode and Display Results ---
    predicted_label = decode_prediction_to_depth_map(prediction_vector)
    
    print(f"\n--- Results for Sample #{SAMPLE_INDEX} at QP={QP_TO_INSPECT} ---")
    print(f"Ground Truth Label: {ground_truth_label}")
    print(f"Predicted Label:    {predicted_label}")
    
    # --- Create Side-by-Side Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(image_patch, cmap='gray')
    draw_cu_partitions(ax1, ground_truth_label, x_offset=0, y_offset=0, color='yellow')
    ax1.set_title("Ground Truth Partitions")
    ax1.axis('off')

    ax2.imshow(image_patch, cmap='gray')
    draw_cu_partitions(ax2, predicted_label, x_offset=0, y_offset=0, color='cyan')
    ax2.set_title("ViT Model Prediction")
    ax2.axis('off')
    
    output_filename = f'vit_pred_sample_{SAMPLE_INDEX}_qp_{QP_TO_INSPECT}.png'
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"\n✅ Comparison visualization saved as '{output_filename}'")

if __name__ == "__main__":
    main()