# Copyright 2020 Bold Hearts
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
from rclpy.node import Node
import numpy as np

from vision_msgs.msg import BoundingBox2D, Detection2DArray


def _bbox_area(bbox: BoundingBox2D) -> int:
    return bbox.size_x * bbox.size_y


def _iou(bbox1: BoundingBox2D, bbox2: BoundingBox2D) -> float:
    bbox1_left = bbox1.center.x - bbox1.size_x / 2
    bbox1_right = bbox1.center.x + bbox1.size_x / 2
    bbox1_top = bbox1.center.y - bbox1.size_y / 2
    bbox1_bottom = bbox1.center.y + bbox1.size_y / 2

    bbox2_left = bbox2.center.x - bbox2.size_x / 2
    bbox2_right = bbox2.center.x + bbox2.size_x / 2
    bbox2_top = bbox2.center.y - bbox2.size_y / 2
    bbox2_bottom = bbox2.center.y + bbox2.size_y / 2

    dx = min(bbox1_right, bbox2_right) - max(bbox1_left, bbox2_left)
    dy = min(bbox1_bottom, bbox2_bottom) - max(bbox1_top, bbox2_top)
    if dx < 0 or dy < 0:
        return 0

    intersection = dx * dy
    union = _bbox_area(bbox1) + _bbox_area(bbox2) - 2 * intersection

    return intersection / union


def _suppress(objects: Detection2DArray, iou_threshold: float = 0.4):
    to_check = list(
        reversed(sorted(objects.detections, key=lambda d: d.results[0].score))
    )

    selected = []
    while len(to_check):
        best_detection = to_check[0]

        # Current best has not been suppressed, so gets selected
        selected.append(best_detection)

        # Add to suppressed list so it will be removed from objects
        # still to check
        suppressed = [best_detection]

        # Go through rest and suppress those with high enough overlap
        for candidate in to_check[1:]:
            # Test only detections of the same class
            if candidate.results[0].id != best_detection.results[0].id:
                continue

            # Determine IoU
            iou = _iou(best_detection.bbox, candidate.bbox)
            if iou >= iou_threshold:
                suppressed.append(candidate)

        # Remove those that are suppressed and continue
        to_check = [d for d in to_check if d not in suppressed]

    # Create output message
    selected_objects = Detection2DArray()
    selected_objects.header = objects.header
    selected_objects.detections = selected

    return selected_objects


def main(args=None):
    rclpy.init(args=args)

    node = Node("non_maximum_suppression", namespace="tflite")

    pub = node.create_publisher(
        Detection2DArray, "objects_nms", rclpy.qos.qos_profile_sensor_data,
    )

    sub = node.create_subscription(
        Detection2DArray,
        "objects",
        lambda msg: pub.publish(_suppress(msg)),
        rclpy.qos.qos_profile_sensor_data,
    )

    rclpy.spin(node)


if __name__ == "__main__":
    main()
