"""Launch classiciation example with MobileNet v2 and camera input."""
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

    video_device = launch.substitutions.LaunchConfiguration("video_device")

    return launch.LaunchDescription(
        [
            launch.actions.DeclareLaunchArgument(
                "video_device",
                default_value=["/dev/video0"],
                description="Which video device to use as camera",
            ),
            launch_ros.actions.Node(
                package="tflite",
                executable="classification_node",
                output="screen",
                parameters=[
                    {"model_path": model, "input_offset": 127.5, "input_scale": 127.5}
                ],
            ),
            launch_ros.actions.Node(
                package="v4l2_camera",
                executable="v4l2_camera_node",
                output="screen",
                parameters=[{"video_device": video_device}],
            ),
            launch_ros.actions.Node(
                package="tflite_util",
                executable="top_classification",
                output="screen",
                parameters=[{"n_top": 3, "labels_file": labels_file}],
            ),
        ]
    )
