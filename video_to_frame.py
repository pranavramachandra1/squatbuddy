import cv2
import pandas as pd
import os
import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
tf.config.run_functions_eagerly(True)

from matplotlib import pyplot as plt
from tqdm import tqdm

from squat_buddy import SquatBuddy

def write_points_to_plot(in_dir: str, out_dir: str):
    sb = SquatBuddy()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    in_paths, out_paths = [os.path.join(in_dir, p) for p in os.listdir(in_dir)], [os.path.join(out_dir, p) for p in os.listdir(in_dir)]
    input_size = 192

    for in_p, out_p in tqdm(zip(in_paths, out_paths), total = len(in_paths)):
        image_path = in_p
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image)

        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        input_image = tf.expand_dims(image, axis=0)
        input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

        # Run model inference.
        keypoints_with_scores = sb.process_image(image_path)

        # Create a blank image the same size as the original
        blank_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        # Visualize the predictions with blank image.
        display_image = tf.expand_dims(blank_image, axis=0)
        display_image = tf.cast(tf.image.resize_with_pad(display_image, 1280, 1280), dtype=tf.int32)
        output_overlay = sb.draw_prediction_on_image(np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)

        # Create the plot with smaller margins and no white borders
        fig = plt.figure(figsize=(5, 5), dpi=300)
        plt.imshow(output_overlay)
        plt.axis('off')

        # Adjust layout to remove white space and save the image tightly around the keypoints
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(out_p, format="png", dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    return

write_points_to_plot(in_dir = "images", out_dir="images_model_results")