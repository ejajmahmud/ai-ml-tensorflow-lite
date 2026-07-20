#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p $DIR/models/

pushd $DIR/models/

## Semantic segmentation
# quantized
curl -L -O https://github.com/google-coral/edgetpu/raw/master/test_data/deeplabv3_mnv2_pascal_quant.tflite
# quantized + Edge optimized
curl -L -O https://github.com/google-coral/edgetpu/raw/master/test_data/deeplabv3_mnv2_pascal_quant_edgetpu.tflite

## Classification
# quantized
curl -L -O https://github.com/google-coral/edgetpu/raw/master/test_data/mobilenet_v2_1.0_224_quant.tflite
# quantized + Edge optimized
curl -L -O https://github.com/google-coral/edgetpu/raw/master/test_data/mobilenet_v2_1.0_224_quant_edgetpu.tflite
# mobilenet classification labels
curl -L -O https://raw.githubusercontent.com/google-coral/edgetpu/master/test_data/imagenet_labels.txt

## Object detection
# quantized
curl -L -O https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess.tflite
# quantized + Edge optimized
curl -L -O https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
# labels
curl -L -O https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-paper.txt
curl -L -O https://raw.githubusercontent.com/google-coral/edgetpu/master/test_data/coco_labels.txt

popd
