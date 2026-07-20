# Copyright 2019 Bold Hearts
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
import numpy as np
from sensor_msgs.msg import Image


cmap = np.array(
    [
        [0, 0, 0],
        [31, 119, 180],
        [255, 127, 14],
        [44, 160, 44],
        [214, 39, 40],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [127, 127, 127],
        [188, 189, 34],
        [23, 190, 207],
        [174, 199, 232],
        [255, 187, 120],
        [152, 223, 138],
        [255, 152, 150],
        [197, 176, 213],
        [196, 156, 148],
        [247, 182, 210],
        [199, 199, 199],
        [219, 219, 141],
        [158, 218, 229],
    ]
)


def _color_segmentation(img: Image):
    if img.encoding.startswith("32FC"):
        n_input_channels = int(img.encoding[4:])
    else:
        raise NotImplementedError(f"Input encoding not handled: {img.encoding}")

    # Create output image
    output_image = Image()
    output_image.header = img.header
    output_image.width = img.width
    output_image.height = img.height
    output_image.step = output_image.width * 3
    output_image.encoding = "rgb8"

    # Create array mapping the image data bytes array
    data = np.frombuffer(img.data, np.float32).reshape(
        img.height, img.width, n_input_channels
    )

    if n_input_channels > 1:
        max_classes = np.argmax(data, axis=2)
        colored_data = cmap[max_classes]
    else:
        # If there is only one channel, assume it already contains the classes
        print(f"max class: {data.max()}")
        colored_data = cmap[data.squeeze().astype(int)]

    output_data = colored_data.ravel()
    output_image.data = output_data.data
    return output_image


def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node("color_segmentation", namespace="tflite")

    qos_profile_name = node.declare_parameter("qos_profile", "sensor_data")
    qos_profile = rclpy.qos.QoSPresetProfiles.get_from_short_key(qos_profile_name.value)

    output_pub = node.create_publisher(Image, "semantic_color", qos_profile)

    node.create_subscription(
        Image,
        "semantic",
        lambda msg: output_pub.publish(_color_segmentation(msg)),
        rclpy.qos.qos_profile_sensor_data,
    )

    rclpy.spin(node)


if __name__ == "__main__":
    main()
