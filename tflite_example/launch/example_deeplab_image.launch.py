"""Launch semantic segmentation example with DeepLabv3 and image input."""
import os

from ament_index_python.packages import get_package_share_directory

import launch

import launch_ros.actions


def generate_launch_description():
    image_path = os.path.join(
        get_package_share_directory("tflite_example"), "images", "dogcat.jpg"
    )

    model = os.path.join(
        get_package_share_directory("tflite_example"),
        "models",
        "deeplabv3_mnv2_pascal_quant.tflite",
    )

    if not os.path.exists(model):
        raise RuntimeError(
            "The deeplabv3 model is not available; expected at: "
            f"{model}, "
            "run `download_models.sh` to fetch it (README: https://gitlab.com/boldhearts/ros2_tflite)."
        )

    rqt_perspective_file = os.path.join(
        get_package_share_directory("tflite_example"),
        "rqt",
        "semantic_segmentation.perspective",
    )

    return launch.LaunchDescription(
        [
            launch_ros.actions.Node(
                package="tflite",
                executable="semanticsegmentation_node",
                output="screen",
                parameters=[
                    {"model_path": model, "input_offset": 128.0, "input_scale": 128.0}
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
                executable="color_segmentation",
                output="screen",
                parameters=[{"qos_profile": "system_default"}],
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
