import os
from squat_buddy import SquatBuddy

sb = SquatBuddy()

video_paths = [
    "test_images/IMG_2422.MOV",
    "test_images/IMG_2507.MOV",
    "test_images/IMG_2547.MOV",
    "test_images/IMG_2818.MOV"
]

img_dir, output_img_dir = "input_images", "output_images"

# Create all the images
for video in video_paths:
    sb.video_to_frame(path = video, dir = img_dir)

sb.write_points_to_plot(in_dir=img_dir, out_dir=output_img_dir)