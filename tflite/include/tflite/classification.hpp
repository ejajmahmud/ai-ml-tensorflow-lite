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

#ifndef TFLITE__CLASSIFICATION_HPP_
#define TFLITE__CLASSIFICATION_HPP_

#include <vision_msgs/msg/classification2_d.hpp>
#include <vector>

#include "visionnode.hpp"


namespace tflite
{

class Classification : public VisionNode
{
public:
  Classification();
  explicit Classification(const rclcpp::NodeOptions & options);

private:
  rclcpp::Publisher<vision_msgs::msg::Classification2D>::SharedPtr output_pub_;

  void preparePub();

  void handleOutput(
    sensor_msgs::msg::Image const & input,
    std::vector<TfLiteTensor const *> const & output_tensors,
    std::vector<std::vector<int>> const & output_dims,
    std::vector<size_t> const & output_sizes) override;
};

}  // namespace tflite

#endif  // TFLITE__CLASSIFICATION_HPP_
