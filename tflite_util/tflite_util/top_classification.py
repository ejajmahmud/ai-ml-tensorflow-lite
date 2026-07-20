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

from typing import List, Optional

import rclpy
import numpy as np
from vision_msgs.msg import Classification2D


def _top_classification(
    msg: Classification2D, n_top: int = 1, labels: Optional[List[str]] = None
) -> Classification2D:
    probability = [h.score for h in msg.results]

    # Get indices of highest scores
    i_sort = np.flip(np.argsort(probability)[-n_top:])

    out_msg = Classification2D()
    out_msg.header = msg.header
    out_msg.results = [msg.results[i] for i in i_sort]
    out_msg.source_img = msg.source_img

    return out_msg


def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node("top_classification", namespace="tflite")

    n_top = node.declare_parameter("n_top")
    labels_file = node.declare_parameter("labels_file")

    node.get_logger().info(f"Labels file: {labels_file.value}")
    if labels_file.value is not None:
        with open(labels_file.value, "r") as f:
            labels = f.read().split("\n")
    else:
        labels = None

    output_pub = node.create_publisher(
        Classification2D, "classes_top", rclpy.qos.qos_profile_sensor_data,
    )

    node.create_subscription(
        Classification2D,
        "classes",
        lambda msg: output_pub.publish(
            _top_classification(msg, n_top=n_top.value, labels=labels)
        ),
        rclpy.qos.qos_profile_sensor_data,
    )

    rclpy.spin(node)


if __name__ == "__main__":
    main()
