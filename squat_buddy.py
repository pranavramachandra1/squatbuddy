import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import cv2

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Some modules to display an animation using imageio.
import imageio
from IPython.display import HTML, display

from constants import KEYPOINT_EDGE_INDS_TO_COLOR
from body_data import BodyData

input_size = 192

class SquatBuddy:

    def __init__(self):
        self.model = tf.lite.Interpreter(model_path='model.tflite')
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

        # Model status
        self.is_loading = False


    
    def movenet(self, input_image):
        """Runs detection on an input image.

        Args:
        input_image: A [1, height, width, 3] tensor represents the input image
            pixels. Note that the height/width should already be resized and match the
            expected input resolution of the model before passing into this function.

        Returns:
        A [1, 1, 17, 3] float numpy array representing the predicted keypoint
        coordinates and scores.
        """
        model = self.module.signatures['serving_default']

        # SavedModel format expects tensor type of int32.
        input_image = tf.cast(input_image, dtype=tf.int32)
        # Run model inference.
        outputs = model(input_image)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints_with_scores = outputs['output_0'].numpy()
        return keypoints_with_scores
    
    def _keypoints_and_edges_for_display(self, keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.11):
        """Returns high confidence keypoints and edges for visualization.

        Args:
            keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.
            height: height of the image in pixels.
            width: width of the image in pixels.
            keypoint_threshold: minimum confidence score for a keypoint to be
            visualized.

        Returns:
            A (keypoints_xy, edges_xy, edge_colors) containing:
            * the coordinates of all keypoints of all detected entities;
            * the coordinates of all skeleton edges of all detected entities;
            * the colors in which the edges should be plotted.
        """
        keypoints_all = []
        keypoint_edges_all = []
        edge_colors = []
        num_instances, _, _, _ = keypoints_with_scores.shape
        for idx in range(num_instances):
            kpts_x = keypoints_with_scores[0, idx, :, 1]
            kpts_y = keypoints_with_scores[0, idx, :, 0]
            kpts_scores = keypoints_with_scores[0, idx, :, 2]
            kpts_absolute_xy = np.stack(
                [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
            kpts_above_thresh_absolute = kpts_absolute_xy[
                kpts_scores > keypoint_threshold, :]
            keypoints_all.append(kpts_above_thresh_absolute)

            for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
                if (kpts_scores[edge_pair[0]] > keypoint_threshold and
                    kpts_scores[edge_pair[1]] > keypoint_threshold):
                    x_start = kpts_absolute_xy[edge_pair[0], 0]
                    y_start = kpts_absolute_xy[edge_pair[0], 1]
                    x_end = kpts_absolute_xy[edge_pair[1], 0]
                    y_end = kpts_absolute_xy[edge_pair[1], 1]
                    line_seg = np.array([[x_start, y_start], [x_end, y_end]])
                    keypoint_edges_all.append(line_seg)
                    edge_colors.append(color)
        if keypoints_all:
            keypoints_xy = np.concatenate(keypoints_all, axis=0)
        else:
            keypoints_xy = np.zeros((0, 17, 2))

        if keypoint_edges_all:
            edges_xy = np.stack(keypoint_edges_all, axis=0)
        else:
            edges_xy = np.zeros((0, 2, 2))
        return keypoints_xy, edges_xy, edge_colors
    def draw_prediction_on_image(
        self, image, keypoints_with_scores, crop_region=None, close_figure=False,
        output_image_height=None):
        """Draws the keypoint predictions on image.

        Args:
            image: A numpy array with shape [height, width, channel] representing the
            pixel values of the input image.
            keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.
            crop_region: A dictionary that defines the coordinates of the bounding box
            of the crop region in normalized coordinates (see the init_crop_region
            function below for more detail). If provided, this function will also
            draw the bounding box on the image.
            output_image_height: An integer indicating the height of the output image.
            Note that the image aspect ratio will be the same as the input image.

        Returns:
            A numpy array with shape [out_height, out_width, channel] representing the
            image overlaid with keypoint predictions.
        """
        height, width, channel = image.shape
        aspect_ratio = float(width) / height
        fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
        # To remove the huge white borders
        fig.tight_layout(pad=0)
        ax.margins(0)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.axis('off')

        im = ax.imshow(image)
        line_segments = LineCollection([], linewidths=(4), linestyle='solid')
        ax.add_collection(line_segments)
        # Turn off tick labels
        scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

        (keypoint_locs, keypoint_edges,
        edge_colors) = self._keypoints_and_edges_for_display(
            keypoints_with_scores, height, width)

        line_segments.set_segments(keypoint_edges)
        line_segments.set_color(edge_colors)
        if keypoint_edges.shape[0]:
            line_segments.set_segments(keypoint_edges)
            line_segments.set_color(edge_colors)
        if keypoint_locs.shape[0]:
            scat.set_offsets(keypoint_locs)

        if crop_region is not None:
            xmin = max(crop_region['x_min'] * width, 0.0)
            ymin = max(crop_region['y_min'] * height, 0.0)
            rec_width = min(crop_region['x_max'], 0.99) * width - xmin
            rec_height = min(crop_region['y_max'], 0.99) * height - ymin
            rect = patches.Rectangle(
                (xmin,ymin),rec_width,rec_height,
                linewidth=1,edgecolor='b',facecolor='none')
            ax.add_patch(rect)

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(
            fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        if output_image_height is not None:
            output_image_width = int(output_image_height / height * width)
            image_from_plot = cv2.resize(
                image_from_plot, dsize=(output_image_width, output_image_height),
                interpolation=cv2.INTER_CUBIC)
        return image_from_plot

    def to_gif(self, images, duration):
        """Converts image sequence (4D numpy array) to gif."""
        imageio.mimsave('./animation.gif', images, duration=duration)
        return embed.embed_file('./animation.gif')

    def progress(self, value, max=100):
        return HTML("""
            <progress
                value='{value}'
                max='{max}',
                style='width: 100%'
            >
                {value}
            </progress>
        """.format(value=value, max=max))
    
    def process_video(self, path: str):
        """
        Processes a video with the movenet library

        Args:
            path: (str) path for video file
        
        Output: 
            body_data (BodyData): BodyData object containing all information on BodyPart positions over the course of the video
        
        
        """

        vidcap = cv2.VideoCapture('test_images/IMG_2547.MOV')
        success,image = vidcap.read()

        body_data = BodyData()

        while success:

            # prepare image
            img = tf.image.resize_with_pad(np.expand_dims(image, axis=0), 192,192)
            input_image = tf.cast(img, dtype=tf.int32)

            # Run model inference.
            self.model.set_tensor(self.input_details[0]['index'], np.array(input_image))
            self.model.invoke()
            keypoints_with_scores = self.model.get_tensor(self.output_details[0]['index'])

            # Add scores to entry
            body_data.add_datapoint(keypoints_with_scores=keypoints_with_scores)

            # Load next image
            success,image = vidcap.read()

        return body_data
    
    def process_image(self, path: str):
        """
        Test running inference on one image
        """

        image = cv2.imread(path)

        img = tf.image.resize_with_pad(np.expand_dims(image, axis=0), 192,192)
        input_image = tf.cast(img, dtype=tf.int32)

        # Run model inference.
        self.model.set_tensor(self.input_details[0]['index'], np.array(input_image))
        self.model.invoke()
        keypoints_with_scores = self.model.get_tensor(self.output_details[0]['index'])

        return keypoints_with_scores
    
    def is_same_keypoints(self, kp1, kp2):
        
        if not np.array(kp1).shape == np.array(kp2).shape:
            return False

        kp1, kp2 = kp1[0][0], kp2[0][0]

        for i in range(len(kp1)):
            for el in range(len(kp1[i])):
                if not np.isclose(kp1[i][el], kp2[i][el]):
                    return False
        
        return True




                
