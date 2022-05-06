import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from module import util as util
from models.tf_movenet import tf_movenet

# Load the input image.
print("program start")

# Load the input image.
image_path = 'input_image.jpeg'
image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image)

model = tf_movenet("movenet_singlepose_lightning")

# Resize and pad the image to keep the aspect ratio and fit the expected size.
input_image = tf.expand_dims(image, axis=0)
input_image = tf.image.resize_with_pad(input_image, model.input_size, model.input_size)

# Run model inference.
keypoints_with_scores = model.tflite_detecet(input_image)

print(keypoints_with_scores)

# Visualize the predictions with image.
display_image = tf.expand_dims(image, axis=0)
display_image = tf.cast(tf.image.resize_with_pad(
    display_image, 1280, 1280), dtype=tf.int32)
output_overlay = util.draw_prediction_on_image(
    np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)

plt.figure(figsize=(5, 5))
plt.imshow(output_overlay)
_ = plt.axis('off')
plt.show()
