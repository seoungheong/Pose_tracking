import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from module import util as util
from models.tf_movenet import tf_movenet

import pyrealsense2 as rs
import cv2
import timeit

pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))


found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

align_to = rs.stream.color
align = rs.align(align_to)

# Start streaming
pipeline.start(config)

counts = 0
fps = [0,0,0,0,0,0,0,0,0,0]
avg = 0

cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

#################################################################
#skeleton model
model_name = "movente_singlepose_thunder"

model = tf_movenet(model_name)

#################################################################

try:
    while True:

        start_t = timeit.default_timer()
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())


        ####################################################################################

        keypoints_with_scores = model.tflite_detecet(color_image)

        color_image, points = util.draw_keypoints(color_image, keypoints_with_scores, 0.4)

        #util.draw3Dplot(depth_image, points)
        ####################################################################################

        terminate_t = timeit.default_timer()
        FPS = int(1. / (terminate_t - start_t))

        fps[counts] = FPS

        if counts < 9:
            counts += 1

        else:
            counts = 0
            avg = sum(fps) / 10

        cv2.putText(color_image, str("FPS: %s" % (avg)), (30, 30), 1, 1, (0, 0, 0))

        # Show images
        cv2.imshow('RealSense', color_image)

        # Press esc or 'q' to close the image window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

finally:
    # Stop streaming
    pipeline.stop()