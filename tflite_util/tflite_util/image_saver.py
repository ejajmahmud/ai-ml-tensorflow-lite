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

# Save images to a file. Implements similar functionality as ROS'
# image_pipeline image_saver node, which should probably be used
# instead of this script when that has been ported to ROS 2, pending
# on this PR:
# https://github.com/ros-perception/image_pipeline/pull/374

from sensor_msgs.msg import Image
import imageio
import numpy as np
import rclpy

_image_idx = 0


def _save_image(img: Image, filename_format: str):
    global _image_idx

    if img.encoding not in ["rgb8", "yuv422"]:
        raise NotImplementedError(f"Input encoding not handled: {img.encoding}")

    if img.encoding == "rgb8":
        np_image = np.frombuffer(img.data, np.uint8).reshape(img.height, img.width, 3)
        imageio.imsave(
            filename_format.format(
                sec=img.header.stamp.sec,
                nsec=img.header.stamp.nanosec,
                enc=img.encoding,
                ext="png",
            ),
            np_image,
        )
    elif img.encoding == "yuv422":
        # encoded as YUYVYUYV...
        np_image = np.frombuffer(img.data, np.uint8).reshape(img.height, img.width, 2)

        # Add a third empty channel just to have 3 to be able to save as image
        np_image_3c = np.concatenate(
            (np_image, np.zeros((*np_image.shape[:2], 1))), axis=2
        )

        imageio.imsave(
            filename_format.format(
                sec=img.header.stamp.sec,
                nsec=img.header.stamp.nanosec,
                enc="yuv422",
                ext="png",
            ),
            np_image_3c,
        )

        y = np_image[:, :, 0]
        u = np_image[:, 0::2, 1]
        v = np_image[:, 1::2, 1]
        u = np.repeat(u, 2, axis=1)
        v = np.repeat(v, 2, axis=1)

        yuv = np.dstack([y, u, v])
        imageio.imsave(
            filename_format.format(
                sec=img.header.stamp.sec,
                nsec=img.header.stamp.nanosec,
                enc="yuv444",
                ext="png",
            ),
            yuv,
        )

        m = np.array(
            [
                [1.0, 1.0, 1.0],
                [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                [1.4019975662231445, -0.7141380310058594, 0.00001542569043522235],
            ]
        )

        rgb = np.dot(yuv, m)
        rgb[:, :, 0] -= 179.45477266423404
        rgb[:, :, 1] += 135.45870971679688
        rgb[:, :, 2] -= 226.8183044444304

        imageio.imsave(
            filename_format.format(
                sec=img.header.stamp.sec,
                nsec=img.header.stamp.nanosec,
                enc="rgb8",
                ext="png",
            ),
            rgb,
        )

    _image_idx += 1


def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node("image_saver", namespace="tflite")

    filename_format_param = node.declare_parameter(
        "filename_format", "image-{enc}-{sec:010d}-{nsec:09d}.{ext}"
    )
    filename_format = filename_format_param.value

    node.create_subscription(
        Image, "image", lambda msg: _save_image(msg, filename_format), 10
    )

    rclpy.spin(node)


if __name__ == "__main__":
    main()
