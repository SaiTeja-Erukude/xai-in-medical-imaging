import os
import numpy as np
import tensorflow as tf
import cv2
from keras.models import load_model

# Function to generate Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    densenet_base = model.get_layer("densenet121")

    # Top layers after base
    gap_layer = model.layers[1]
    dense1 = model.layers[2]
    bn = model.layers[3]
    drop1 = model.layers[4]
    dense2 = model.layers[5]
    drop2 = model.layers[6]
    final_dense = model.layers[7]

    def top_forward_pass(feature_maps):
        x = gap_layer(feature_maps)
        x = dense1(x)
        x = bn(x)
        x = drop1(x)
        x = dense2(x)
        x = drop2(x)
        x = final_dense(x)
        return x

    grad_model = tf.keras.models.Model(
        [densenet_base.input],
        [densenet_base.get_layer(last_conv_layer_name).output, densenet_base.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = grad_model(img_array)
        predictions = top_forward_pass(base_outputs)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Function to superimpose heatmap and save
def save_and_display_gradcam(img_path, heatmap, output_path, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (WIDTH, HEIGHT))
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap_color * alpha + img
    
    cv2.imwrite(output_path, superimposed_img)

# Configuration
MODEL_PATH = "D:/Projects/XAI in Medical Imaging/models/densenet/densenet121_chest.h5"
TEST_DIR   = "D:/Projects/XAI in Medical Imaging/data/chest/test"
HEIGHT, WIDTH = 256, 256
OUTPUT_DIR = "D:/Projects/XAI in Medical Imaging/predictions/chest/densenet/grad_cam"
last_conv_layer_name = "conv5_block16_concat"

# Load model
print("Loading trained DenseNet121 model...")
model = load_model(MODEL_PATH)

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Walk through test directory and apply Grad-CAM
for root, _, files in os.walk(TEST_DIR):
    for fname in files:
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(root, fname)
            rel_path = os.path.relpath(img_path, TEST_DIR)
            output_path = os.path.join(OUTPUT_DIR, rel_path)

            # Create output subfolder if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            try:
                # Load and preprocess image
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(HEIGHT, WIDTH))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                # Generate heatmap
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

                # Save overlay
                save_and_display_gradcam(img_path, heatmap, output_path)

                print(f"Saved Grad-CAM for {rel_path} → {output_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
