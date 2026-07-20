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

#include "tflite/classification.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace tflite
{

Classification::Classification()
: VisionNode{"classification"}
{
  preparePub();
}

Classification::Classification(const rclcpp::NodeOptions & options)
: VisionNode{"classification", options}
{
  preparePub();
}

void Classification::preparePub()
{
  output_pub_ = create_publisher<vision_msgs::msg::Classification2D>(
    "classes",
    rclcpp::SensorDataQoS{}
  );
}

void Classification::handleOutput(
  sensor_msgs::msg::Image const & input,
  std::vector<TfLiteTensor const *> const & output_tensors,
  std::vector<std::vector<int>> const & output_dims,
  std::vector<size_t> const & output_sizes)
{
  (void)output_sizes;

  // Get output tensor data
  // Only one output tensor expected for classification
  auto out_msg = std::make_unique<vision_msgs::msg::Classification2D>();
  out_msg->header = input.header;

  // Last dimesnion of output is number of classes
  auto nClasses = output_dims[0].back();
  out_msg->results.resize(nClasses);

  if (output_tensors[0]->type == kTfLiteUInt8) {
    auto * output = output_tensors[0]->data.uint8;
    for (auto i = 0; i < nClasses; ++i) {
      out_msg->results[i].id = std::to_string(i);
      out_msg->results[i].score = output[i] / 255.0;
    }
  } else if (output_tensors[0]->type == kTfLiteFloat32) {
    auto * output = output_tensors[0]->data.f;
    for (auto i = 0; i < nClasses; ++i) {
      out_msg->results[i].id = std::to_string(i);
      out_msg->results[i].score = output[i];
    }
  } else {
    RCLCPP_ERROR_STREAM_ONCE(
      get_logger(),
      "Unsupported output type: " << TfLiteTypeGetName(output_tensors[0]->type));
    return;
  }

  output_pub_->publish(std::move(out_msg));
}

}  // namespace tflite

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(tflite::Classification)
