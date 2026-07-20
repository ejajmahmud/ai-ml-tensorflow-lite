from setuptools import setup

package_name = 'tflite_util'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
    ],
    install_requires=['setuptools', 'imageio', 'numpy'],
    maintainer='Sander van Dijk',
    maintainer_email='sgvandijk@gmail.com',
    keywords=['ROS'],
    description='Utilities for working with TfLite for ROS 2',
    license='Apache License, Version 2.0',
    entry_points={
        'console_scripts': [
            'non_maximum_suppression = tflite_util.non_maximum_suppression:main',
            'draw_detections = tflite_util.draw_detections:main',
            'color_segmentation = tflite_util.color_segmentation:main',
            'top_classification = tflite_util.top_classification:main',
            'extract_channel = tflite_util.extract_channel:main',
            'image_saver = tflite_util.image_saver:main',
        ],
    },
)
