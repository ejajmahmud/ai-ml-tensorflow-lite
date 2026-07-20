// Copyright 2020 Bold Hearts
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tflite/objectdetection.hpp"

#include <vision_msgs/create_aabb.hpp>
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>


namespace tflite
{

ObjectDetection::ObjectDetection()
: VisionNode{"object_detection"}
{
  preparePub();
}

ObjectDetection::ObjectDetection(const rclcpp::NodeOptions & options)
: VisionNode{"object_detection", options}
{
  preparePub();
}


void ObjectDetection::preparePub()
{
  output_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>(
    "objects",
    rclcpp::SensorDataQoS{}
  );
}

void ObjectDetection::handleOutput(
  sensor_msgs::msg::Image const & input,
  std::vector<TfLiteTensor const *> const & output_tensors,
  std::vector<std::vector<int>> const & output_dims,
  std::vector<size_t> const & output_sizes)
{
  (void)output_dims;
  (void)output_sizes;

  // Get output tensor data
  // 4 output tensors expected:
  // - locations - 4d: y top, x left, y bottom, x right (normalized 0-1)
  // - classes - Nx1d where N is max number of detections
  // - scores - Nx1d where N is max number of detections
  // - num detections - 1x1d, actual number of detections
  // TODO(sgvandijk): this is for SSD type detectors
  auto out_msg = std::make_unique<vision_msgs::msg::Detection2DArray>();
  out_msg->header = input.header;

  // We assume all outputs have the same type
  if (output_tensors[0]->type == kTfLiteFloat32) {
    auto * locations = reinterpret_cast<float(*)[4]>(output_tensors[0]->data.f);
    auto * classes = output_tensors[1]->data.f;
    auto * scores = output_tensors[2]->data.f;
    auto * num = output_tensors[3]->data.f;

    out_msg->detections.resize(num[0]);
    for (auto i = 0u; i < num[0]; ++i) {
      auto & detection = out_msg->detections[i];

      // Box outer coordinates in pixels
      auto top = std::max(0.0f, locations[i][0]) * input.height;
      auto left = std::max(0.0f, locations[i][1]) * input.width;
      auto bottom = std::min(1.0f, locations[i][2]) * input.height;
      auto right = std::min(1.0f, locations[i][3]) * input.width;

      detection.bbox = vision_msgs::createAABB2D(left, top, right - left, bottom - top);

      // Currently only working with networks that output a single class + score
      detection.results.resize(1);
      detection.results[0].id = std::to_string(static_cast<int>(classes[i]));
      detection.results[0].score = scores[i];
    }
  }

  output_pub_->publish(std::move(out_msg));
}

}  // namespace tflite

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(tflite::ObjectDetection)
