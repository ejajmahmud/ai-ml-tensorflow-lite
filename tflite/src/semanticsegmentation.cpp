// Copyright 2019 Bold Hearts
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

#include "tflite/semanticsegmentation.hpp"

// #include <tensorflow/lite/delegates/gpu/gl_delegate.h>

#include <rclcpp/logging.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef HAVE_EDGETPU
#include <edgetpu_c.h>
#endif

namespace tflite
{

SemanticSegmentation::SemanticSegmentation()
: VisionNode{"semantic_segmentation"}
{
  preparePub();
}

SemanticSegmentation::SemanticSegmentation(const rclcpp::NodeOptions & options)
: VisionNode{"semantic_segmentation", options}
{
  preparePub();
}

void SemanticSegmentation::preparePub()
{
  output_pub_ = create_publisher<sensor_msgs::msg::Image>("semantic", rclcpp::SensorDataQoS{});
}

void SemanticSegmentation::handleOutput(
  sensor_msgs::msg::Image const & input,
  std::vector<TfLiteTensor const *> const & output_tensors,
  std::vector<std::vector<int>> const & output_dims,
  std::vector<size_t> const & output_sizes)
{
  // Get output tensor data
  // Only one output tensor expected for semantic segmentation
  auto out_msg = std::make_unique<sensor_msgs::msg::Image>();
  out_msg->header = input.header;
  out_msg->height = output_dims[0][1];
  out_msg->width = output_dims[0][2];
  auto channels = output_dims[0].size() >= 4 ? output_dims[0][3] : 1;
  out_msg->step = out_msg->width * channels * sizeof(float);
  // TODO(sgvandijk): use output type of network instead of transforming to float
  out_msg->data = sensor_msgs::msg::Image::_data_type(
    out_msg->width * out_msg->height * channels * sizeof(float));
  out_msg->encoding = std::string{"32FC"} + std::to_string(channels);

  if (output_tensors[0]->type == kTfLiteInt64) {
    auto * output = output_tensors[0]->data.i64;
    std::copy(
      output, output + output_sizes[0] / sizeof(int64_t),
      reinterpret_cast<float *>(&out_msg->data[0])
    );
  } else if (output_tensors[0]->type == kTfLiteFloat32) {
    auto * output = output_tensors[0]->data.f;
    std::copy(
      output, output + output_sizes[0] / sizeof(float),
      reinterpret_cast<float *>(&out_msg->data[0])
    );
  }

  // Resize output image if needed
  if (out_msg->height != input.height || out_msg->width != input.width) {
    RCLCPP_WARN_ONCE(get_logger(), "Image size does not match model output; resizing");
    out_msg = resize(*out_msg, input.width, input.height);
  }

  output_pub_->publish(std::move(out_msg));
}

}  // namespace tflite

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(tflite::SemanticSegmentation)
