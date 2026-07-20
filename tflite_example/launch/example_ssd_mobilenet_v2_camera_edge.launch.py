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
        "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
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

    log_level = launch.substitutions.LaunchConfiguration("log_level")
    video_device = launch.substitutions.LaunchConfiguration("video_device")

    return launch.LaunchDescription(
        [
            launch.actions.DeclareLaunchArgument(
                "log_level",
                default_value=["INFO"],
                description="Log level to use for all nodes",
            ),
            launch.actions.DeclareLaunchArgument(
                "video_device",
                default_value=["/dev/video0"],
                description="Which video device to use as camera",
            ),
            launch_ros.actions.Node(
                package="tflite",
                executable="objectdetection_node",
                output="screen",
                parameters=[
                    {
                        "model_path": model,
                        "input_offset": 127.5,
                        "input_scale": 127.5,
                        "delegate": "edgetpu",
                    }
                ],
                arguments=["--ros-args", "--log-level", log_level],
            ),
            launch_ros.actions.Node(
                package="v4l2_camera",
                executable="v4l2_camera_node",
                output="screen",
                parameters=[{"video_device": video_device}],
                arguments=["--ros-args", "--log-level", log_level],
            ),
            launch_ros.actions.Node(
                package="tflite_util",
                executable="draw_detections",
                output="screen",
                parameters=[
                    {"labels_file": labels_file, "qos_profile": "system_default"}
                ],
                arguments=["--ros-args", "--log-level", log_level],
            ),
            launch_ros.actions.Node(
                package="tflite_util",
                executable="non_maximum_suppression",
                output="screen",
                arguments=["--ros-args", "--log-level", log_level],
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
                arguments=["--ros-args", "--log-level", log_level],
            ),
        ]
    )
