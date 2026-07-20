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

#ifndef TFLITE__VISIONNODE_HPP_
#define TFLITE__VISIONNODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <tensorflow/lite/model.h>

#include <memory>
#include <string>
#include <vector>

namespace tflite
{

class VisionNode : public rclcpp::Node
{
public:
  explicit VisionNode(const std::string & node_name);
  VisionNode(const std::string & node_name, const rclcpp::NodeOptions & options);

protected:
  std::unique_ptr<sensor_msgs::msg::Image> resize(
    sensor_msgs::msg::Image const & img,
    int width, int height);

private:
  std::unique_ptr<FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;

  TfLiteTensor const * input_tensor_;
  std::vector<TfLiteTensor const *> output_tensors_;

  std::vector<int> input_dims_;
  std::vector<std::vector<int>> output_dims_;
  std::vector<size_t> output_sizes_;

  float input_offset_;
  float input_scale_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

  virtual void handleOutput(
    sensor_msgs::msg::Image const & input,
    std::vector<TfLiteTensor const *> const & output_tensors,
    std::vector<std::vector<int>> const & output_dims,
    std::vector<size_t> const & output_sizes) = 0;

  void loadModel();
  void prepareSub();
};

}  // namespace tflite

#endif  // TFLITE__VISIONNODE_HPP_
