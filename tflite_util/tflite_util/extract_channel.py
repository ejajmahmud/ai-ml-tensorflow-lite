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


def _extract_channel(img: Image, channel_idx: int):
    if img.encoding.startswith("8UC"):
        n_input_channels = int(img.encoding[3:])
    elif img.encoding == "yuv422":
        n_input_channels = 2
    else:
        n_input_channels = 3
        # raise NotImplementedError(
        #     f"Input encoding not handled: {img.encoding}"
        # )

    if channel_idx >= n_input_channels:
        raise ValueError(
            f"Channel index out of range: {channel_idx}; "
            f"number of channels: {n_input_channels}"
        )

    output_image = Image()
    output_image.width = img.width
    output_image.height = img.height
    output_image.step = output_image.width
    output_image.encoding = "mono8"

    data = np.frombuffer(img.data, np.uint8).reshape(
        img.height, img.width, n_input_channels
    )
    data = data[:, :, channel_idx].ravel()
    data = data.data

    output_image.data = data
    return output_image


def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node("extract_channel", namespace="tflite")

    channel_idx_param = node.declare_parameter("channel_idx", 0)
    channel_idx = channel_idx_param.value

    output_pub = node.create_publisher(Image, "extracted_channel", 10)

    node.create_subscription(
        Image,
        "extract_input",
        lambda msg: output_pub.publish(_extract_channel(msg, channel_idx)),
        10
    )

    rclpy.spin(node)


if __name__ == '__main__':
    main()
