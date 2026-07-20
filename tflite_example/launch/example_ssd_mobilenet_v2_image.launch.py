"""Launch classiciation example with SSD MobileNet v2 and image input."""
import os

from ament_index_python.packages import get_package_share_directory

import launch

import launch_ros.actions


def generate_launch_description():
    image_path = os.path.join(
        get_package_share_directory("tflite_example"), "images", "dogumbrella.jpg"
    )

    model = os.path.join(
        get_package_share_directory("tflite_example"),
        "models",
        "ssd_mobilenet_v2_coco_quant_postprocess.tflite",
    )

    if not os.path.exists(model):
        raise RuntimeError(
            "The ssd_mobilenet_v2 model is not available; expected at: "
            f"{model}, "
            "run `download_models.sh` to fetch it (README: https://gitlab.com/boldhearts/ros2_tflite)."
        )

    labels_file = os.path.join(
        get_package_share_directory("tflite_example"),
        "models",
        "coco-labels-paper.txt",
    )

    rqt_perspective_file = os.path.join(
        get_package_share_directory("tflite_example"),
        "rqt",
        "object_detection.perspective",
    )

    return launch.LaunchDescription(
        [
            launch_ros.actions.Node(
                package="tflite",
                executable="objectdetection_node",
                output="screen",
                parameters=[
                    {"model_path": model, "input_offset": 127.5, "input_scale": 127.5}
                ],
            ),
            launch_ros.actions.Node(
                package="image_publisher",
                executable="image_publisher_node",
                output="screen",
                arguments=[image_path],
            ),
            launch_ros.actions.Node(
                package="tflite_util",
                executable="draw_detections",
                output="screen",
                parameters=[
                    {"labels_file": labels_file, "qos_profile": "system_default"}
                ],
            ),
            launch_ros.actions.Node(
                package="tflite_util",
                executable="non_maximum_suppression",
                output="screen",
            ),
            launch_ros.actions.Node(
                package="tflite_util",
                executable="draw_detections",
                name="draw_nms_detections",
                output="screen",
                parameters=[
                    {"labels_file": labels_file, "qos_profile": "system_default"}
                ],
                remappings=[
                    ("/tflite/objects", "/tflite/objects_nms"),
                    ("/tflite/objects_drawn", "/tflite/objects_nms_drawn"),
                ],
            ),
            launch_ros.actions.Node(
                package="rqt_gui",
                executable="rqt_gui",
                arguments=["--perspective-file", rqt_perspective_file],
                output="screen",
                on_exit=launch.actions.Shutdown(),
            ),
        ]
    )
