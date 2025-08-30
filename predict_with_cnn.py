import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import data_info as di
import net_CTU64 as nt # Import the network definition

# Enable TensorFlow 1.x compatibility behavior
tf.compat.v1.disable_eager_execution()

# --- 1. Configuration: SET THESE VALUES ---

# Paths to your data and model
YUV_PATH = '/workspaces/HEVC_CPH_Data_Visualization/yuv'
INFO_PATH = '/workspaces/HEVC_CPH_Data_Visualization/info'
# Path to the TensorFlow model checkpoint (provide the base name)
MODEL_CHECKPOINT_PATH = '/workspaces/HEVC_CPH_Data_Visualization/Models/model_20181227_060655_1000000_qp22.dat'

# Details of the frame to inspect
VIDEO_FILENAME = 'IntraTrain_768x512'
FRAME_TO_VISUALIZE = 7
# The QP must match the model you are loading (e.g., qp22 for the qp22 model)
QP_TO_INSPECT = 22

# The specific pixel coordinate you want to inspect
X_COORDINATE = 128
Y_COORDINATE = 192

# --- 2. Helper Functions ---

def read_YUV420_frame(fid, width, height, frame_index):
    frame_bytes = (width * height * 3) // 2
    fid.seek(frame_index * frame_bytes)
    Y_buf = fid.read(width * height)
    if not Y_buf: return None
    return np.frombuffer(Y_buf, dtype=np.uint8).reshape([height, width])

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
    def draw_rect(x, y, size, color='cyan', lw=1.2):
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

def decode_cnn_prediction_to_depth_map(pred_64, pred_32, pred_16):
    d_64 = (pred_64 > 0.5).astype(int).flatten()
    d_32 = (pred_32 > 0.5).astype(int).flatten()
    d_16 = (pred_16 > 0.5).astype(int).flatten()
    depth_map = np.zeros(16, dtype=int)
    if d_64[0] == 1:
        depth_map[:] = 1
        for i in range(4):
            if d_32[i] == 1:
                start_idx = (i//2)*8 + (i%2)*2
                depth_map[start_idx:start_idx+2] = 2
                depth_map[start_idx+4:start_idx+6] = 2
    for i in range(16):
        if d_16[i] == 1:
            depth_map[i] = 3
    return depth_map

# --- 3. Main Prediction and Visualization Logic ---

def main():
    try:
        video_index = di.YUV_NAME_LIST_FULL.index(VIDEO_FILENAME)
        width, height = di.YUV_WIDTH_LIST_FULL[video_index], di.YUV_HEIGHT_LIST_FULL[video_index]
        yuv_file = os.path.join(YUV_PATH, VIDEO_FILENAME + '.yuv')
        with open(yuv_file, 'rb') as f_yuv:
            frame_pixels = read_YUV420_frame(f_yuv, width, height, FRAME_TO_VISUALIZE)
        if frame_pixels is None:
            print(f"Error: Could not read frame #{FRAME_TO_VISUALIZE}."); return
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading data: {e}"); return

    ctu_x_idx, ctu_y_idx = X_COORDINATE // 64, Y_COORDINATE // 64
    x_start, y_start = ctu_x_idx * 64, ctu_y_idx * 64
    image_patch = frame_pixels[y_start:y_start+64, x_start:x_start+64]
    
    # Normalize image and QP. The CNN model expects a different normalization than the ViT.
    image_tensor = image_patch.astype(np.float32).reshape(1, 64, 64, 1)
    qp_tensor = np.array([float(QP_TO_INSPECT)]).reshape(1, 1)

    print(f"Loading TensorFlow model from '{MODEL_CHECKPOINT_PATH}'...")
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        x = tf.compat.v1.placeholder("float", [None, 64, 64, 1])
        y_ = tf.compat.v1.placeholder("float", [None, 16])
        qp = tf.compat.v1.placeholder("float", [None, 1])
        isdrop = tf.compat.v1.placeholder("float")
        global_step = tf.compat.v1.placeholder("float")
        
        _, _, _, y_conv_64, y_conv_32, y_conv_16, _, _, _, _, _, _ = nt.net(
            x, y_, qp, isdrop, global_step, 0.01, 0.9, 1, 1)

        # --- THIS IS THE FIX ---
        # Get a list of only the model's trainable variables (weights and biases)
        model_variables = tf.compat.v1.trainable_variables()
        # Create a Saver that ONLY restores these variables, ignoring the optimizer state
        saver = tf.compat.v1.train.Saver(var_list=model_variables)
        
        saver.restore(sess, MODEL_CHECKPOINT_PATH)
        print("Model restored successfully.")
        
        pred_64, pred_32, pred_16 = sess.run(
            [y_conv_64, y_conv_32, y_conv_16],
            feed_dict={x: image_tensor, qp: qp_tensor, isdrop: 0.0, global_step: 0.0}
        )

    predicted_label = decode_cnn_prediction_to_depth_map(pred_64, pred_32, pred_16)
    print(f"\nCoordinate ({X_COORDINATE}, {Y_COORDINATE}) in CTU ({ctu_x_idx}, {ctu_y_idx})")
    print(f"Predicted Label: {predicted_label}")

    txt_filename = f'cnn_prediction_X{X_COORDINATE}_Y{Y_COORDINATE}_qp{QP_TO_INSPECT}.txt'
    np.savetxt(txt_filename, predicted_label.reshape(4,4), fmt='%d')
    print(f"✅ Predicted label map saved to '{txt_filename}'")
    
    png_filename = f'cnn_viz_X{X_COORDINATE}_Y{Y_COORDINATE}_qp{QP_TO_INSPECT}.png'
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(image_patch, cmap='gray')
    draw_cu_partitions(ax, predicted_label, x_offset=0, y_offset=0)
    ax.set_title(f"CNN Prediction at ({X_COORDINATE}, {Y_COORDINATE})")
    ax.axis('off')
    plt.savefig(png_filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✅ Partition visualization saved to '{png_filename}'")

if __name__ == "__main__":
    main()