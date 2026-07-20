"""Launch classiciation example with MobileNet v2 and image input."""
import os

from ament_index_python.packages import get_package_share_directory

import launch

import launch_ros.actions


def generate_launch_description():
    image_path = os.path.join(
        get_package_share_directory("tflite_example"), "images", "wombat.jpg"
    )

    model = os.path.join(
        get_package_share_directory("tflite_example"),
        "models",
        "mobilenet_v2_1.0_224_quant.tflite",
    )

    if not os.path.exists(model):
        raise RuntimeError(
            "The mobilenet_v2_1.0_224 model is not available; expected at: "
            f"{model}, "
            "run `download_models.sh` to fetch it (README: https://gitlab.com/boldhearts/ros2_tflite)."
        )

    labels_file = os.path.join(
        get_package_share_directory("tflite_example"),
        "models",
        "imagenet_labels.txt",
    )

    return launch.LaunchDescription(
        [
            launch_ros.actions.Node(
                package="tflite",
                executable="classification_node",
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
                executable="top_classification",
                output="screen",
                parameters=[{"n_top": 3, "labels_file": labels_file}],
            ),
        ]
    )
