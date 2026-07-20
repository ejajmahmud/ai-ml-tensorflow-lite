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
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
import numpy as np
import cv2
from message_filters import Subscriber, TimeSynchronizer

cmap = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (174, 199, 232),
    (255, 187, 120),
    (152, 223, 138),
    (255, 152, 150),
    (197, 176, 213),
    (196, 156, 148),
    (247, 182, 210),
    (199, 199, 199),
    (219, 219, 141),
    (158, 218, 229),
]


class DrawDetections(Node):
    def __init__(self):
        super(DrawDetections, self).__init__("draw_detections", namespace="tflite")

        self.threshold = self.declare_parameter("threshold", 0.4)
        self.n_top = self.declare_parameter("n_top", 20)
        self.labels_file = self.declare_parameter("labels_file")

        qos_profile_name = self.declare_parameter("qos_profile", "sensor_data")
        qos_profile = rclpy.qos.QoSPresetProfiles.get_from_short_key(
            qos_profile_name.value
        )

        if self.labels_file.value is not None:
            with open(self.labels_file.value, "r") as f:
                self.labels = f.read().split("\n")
        else:
            self.labels = None

        self.output_pub = self.create_publisher(Image, "objects_drawn", qos_profile)

        img_sub = Subscriber(
            self, Image, "/image_raw", qos_profile=rclpy.qos.qos_profile_sensor_data
        )
        objects_sub = Subscriber(
            self,
            Detection2DArray,
            "objects",
            qos_profile=rclpy.qos.qos_profile_sensor_data,
        )

        sync = TimeSynchronizer([img_sub, objects_sub], 10)

        sync.registerCallback(self._draw_object_detections)

    def _draw_object_detections(
        self, image: Image, objects: Detection2DArray,
    ):
        if len(objects.detections) == 0:
            return

        # Create output image
        output_image = Image()
        output_image.header = objects.header
        output_image.width = image.width
        output_image.height = image.height
        output_image.step = image.step
        output_image.encoding = image.encoding

        # Map data array to a 3D Numpy array
        output_data = np.frombuffer(image.data, np.uint8).reshape(
            image.height, image.width, 3
        )

        # Detections sorted from highest to lowest score
        ranked_detections = reversed(
            sorted(objects.detections, key=lambda d: d.results[0].score)
        )

        for i, detection in zip(range(self.n_top.value), ranked_detections):
            # Currently only working with networks that output a single class + score
            result = detection.results[0]
            if result.score < self.threshold.value:
                break

            color = cmap[(int(result.id) + 1) % len(cmap)]

            left = int(detection.bbox.center.x - detection.bbox.size_x / 2)
            right = int(detection.bbox.center.x + detection.bbox.size_x / 2)
            top = int(detection.bbox.center.y - detection.bbox.size_y / 2)
            bottom = int(detection.bbox.center.y + detection.bbox.size_y / 2)

            # Draw bounding box
            cv2.rectangle(
                output_data, (left, top), (right, bottom), color, thickness=2,
            )

            # Draw label
            if self.labels:
                label_idx = int(result.id)
                if label_idx >= len(self.labels):
                    raise IndexError(
                        f"label index out of range: {label_idx} >= {len(self.labels)}"
                    )
                label = self.labels[int(result.id)]
            else:
                label = str(result.id)
            text = f"{label} {result.score:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)

            cv2.rectangle(
                output_data,
                (left, top),
                (left + text_size[0], top + text_size[1] + 4),
                color,
                thickness=-1,
            )

            cv2.putText(
                output_data,
                text,
                (left, top + text_size[1]),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )
        output_image.data = output_data.ravel().data

        self.output_pub.publish(output_image)


def main(args=None):
    rclpy.init(args=args)

    node = DrawDetections()

    rclpy.spin(node)


if __name__ == "__main__":
    main()
