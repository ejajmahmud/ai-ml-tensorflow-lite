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

#include "tflite/visionnode.hpp"

#include <tensorflow/lite/kernels/register.h>

#include <cv_bridge/cv_bridge.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef HAVE_EDGETPU
#include <edgetpu_c.h>
#endif

namespace tflite
{

VisionNode::VisionNode(const std::string & node_name)
: rclcpp::Node{node_name, "tflite"}
{
  loadModel();
  prepareSub();
}

VisionNode::VisionNode(const std::string & node_name, const rclcpp::NodeOptions & options)
: rclcpp::Node{node_name, "tflite", options}
{
  loadModel();
  prepareSub();
}

void VisionNode::loadModel()
{
  // Build model from file; the interpreter runs the actual inference
  auto model_path = declare_parameter<std::string>("model_path", "model.tflite");
  model_ = FlatBufferModel::BuildFromFile(model_path.c_str());

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*model_, resolver)(&interpreter_);

  // See if we should use a delegate
  auto delegate = declare_parameter<std::string>("delegate", "none");

  if (delegate == "edgetpu") {
#ifdef HAVE_EDGETPU
    size_t num_devices;
    std::unique_ptr<edgetpu_device, decltype(& edgetpu_free_devices)> devices(
      edgetpu_list_devices(&num_devices), &edgetpu_free_devices);

    if (num_devices > 0) {
      RCLCPP_INFO(get_logger(), "Found Edge TPU device; setting up delegate");
      const auto & device = devices.get()[0];
      auto * delegate = edgetpu_create_delegate(device.type, device.path, nullptr, 0);
      if (interpreter_->ModifyGraphWithDelegate({delegate, edgetpu_free_delegate}) != kTfLiteOk) {
        RCLCPP_ERROR(get_logger(), "Failed modifying graph with Edge TPU delegate");
      }
    } else {
      RCLCPP_WARN(get_logger(), "Compiled with Edge TPU support, but no device found");
    }
#else
    RCLCPP_ERROR(get_logger(), "Edge TPU delegate requested, but not built to support it");
#endif
    // } else if (delegate == "gpu") {
    //   auto options = TfLiteGpuDelegateOptions{};
    //   options.metadata = NULL;
    //   options.compile_options.precision_loss_allowed = 1;  // FP16
    //   options.compile_options.preferred_gl_object_type = TFLITE_GL_OBJECT_TYPE_FASTEST;
    //   options.compile_options.dynamic_batch_enabled = 0;

    //   auto * delegate = TfLiteGpuDelegateCreate(&options);
    //   if (interpreter_->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
    //     RCLCPP_ERROR(get_logger(), "Failed modifying graph with GPU delegate");
    //   }
  } else if (delegate != "none") {
    RCLCPP_WARN_STREAM(get_logger(), "Delegate note supported: " << delegate);
  }

  interpreter_->AllocateTensors();

  // Determine input and output dimensions
  input_tensor_ = interpreter_->tensor(interpreter_->inputs()[0]);
  RCLCPP_INFO_STREAM(get_logger(), "Number of output tensors: " << interpreter_->outputs().size());

  std::copy(
    input_tensor_->dims->data, input_tensor_->dims->data + input_tensor_->dims->size,
    std::back_inserter(input_dims_)
  );

  for (auto o : interpreter_->outputs()) {
    auto const * output_tensor = interpreter_->tensor(o);
    output_tensors_.push_back(output_tensor);

    auto dims = std::vector<int>{};
    std::copy(
      output_tensor->dims->data, output_tensor->dims->data + output_tensor->dims->size,
      std::back_inserter(dims)
    );
    output_dims_.push_back(dims);

    output_sizes_.push_back(output_tensor->bytes);
  }

  // Get other parameters
  input_offset_ = declare_parameter("input_offset", 0.0f);
  input_scale_ = declare_parameter("input_scale", 1.0f);

  // Log details
  RCLCPP_INFO(get_logger(), std::string{"Input name: "} + interpreter_->GetInputName(0));
  RCLCPP_INFO_STREAM(get_logger(), "Input type: " << TfLiteTypeGetName(input_tensor_->type));
  RCLCPP_INFO(get_logger(), std::string{"Input dimensions: "});
  for (auto d : input_dims_) {
    RCLCPP_INFO(get_logger(), std::to_string(d));
  }

  for (auto i = 0u; i < output_tensors_.size(); ++i) {
    RCLCPP_INFO(get_logger(), std::string{"---"});
    RCLCPP_INFO(get_logger(), std::string{"Output name: "} + interpreter_->GetOutputName(0));
    RCLCPP_INFO_STREAM(
      get_logger(),
      "Output type: " << TfLiteTypeGetName(output_tensors_[i]->type));
    RCLCPP_INFO_STREAM(get_logger(), "Output size: " << std::to_string(output_sizes_[i]));
    RCLCPP_INFO(get_logger(), std::string{"Output dimensions: "});
    for (auto d : output_dims_[i]) {
      RCLCPP_INFO(get_logger(), std::to_string(d));
    }
  }
}

void VisionNode::prepareSub()
{
  image_sub_ =
    create_subscription<sensor_msgs::msg::Image>(
    "/image_raw",
    rclcpp::SensorDataQoS{},
    [this](sensor_msgs::msg::Image::UniquePtr img) {
      RCLCPP_DEBUG(get_logger(), "Received image");

      auto ts = std::vector<std::pair<std::string, rclcpp::Time>>{};
      ts.push_back({"t0", now()});
      auto input_img_width = img->width;
      auto input_img_height = img->height;

      auto original_img = sensor_msgs::msg::Image::SharedPtr{std::move(img)};
      auto input_img = sensor_msgs::msg::Image::SharedPtr{};

      // Resize input image if needed
      if (
        int64_t{input_img_height} != input_dims_[1] ||
        int64_t{input_img_width} != input_dims_[2])
      {
        RCLCPP_WARN_ONCE(get_logger(), "Image size does not match model input; resizing");
        input_img = resize(*original_img, input_dims_[2], input_dims_[1]);
      } else {
        input_img = original_img;
      }

      ts.push_back({"resize input", now()});

      // fill input tensor data
      if (input_tensor_->type == kTfLiteUInt8) {
        auto * input = interpreter_->typed_input_tensor<uint8_t>(0);
        std::copy(input_img->data.begin(), input_img->data.end(), input);

      } else if (input_tensor_->type == kTfLiteFloat32) {
        auto * input = interpreter_->typed_input_tensor<float>(0);

        // Scale/offset input if needed
        if (input_offset_ != 0.0f || input_scale_ != 1.0f) {
          std::transform(
            input_img->data.begin(), input_img->data.end(), input, [this](uint8_t v) {
              return (v - input_offset_) / input_scale_;
            });
        } else {
          std::copy(input_img->data.begin(), input_img->data.end(), input);
        }
      } else {
        RCLCPP_ERROR_STREAM_ONCE(
          get_logger(),
          "Unsupported input type: " << TfLiteTypeGetName(input_tensor_->type));
        return;
      }
      ts.push_back({"fill input", now()});

      // Run inference
      interpreter_->Invoke();
      ts.push_back({"invoke", now()});

      handleOutput(*original_img, output_tensors_, output_dims_, output_sizes_);
      ts.push_back({"handle output", now()});
      RCLCPP_DEBUG(get_logger(), "===== ts:");
      for (auto i = 1u; i < ts.size(); ++i) {
        RCLCPP_DEBUG_STREAM(
          get_logger(),
          ts[i].first << ":\t" <<
          (ts[i].second.nanoseconds() - ts[i - 1].second.nanoseconds()) / 1000.0 / 1000.0);
      }
      RCLCPP_DEBUG_STREAM(
        get_logger(),
        "total:\t" <<
        (ts[ts.size() - 1].second.nanoseconds() - ts[0].second.nanoseconds()) / 1000.0 / 1000.0);
    });
}

std::unique_ptr<sensor_msgs::msg::Image> VisionNode::resize(
  sensor_msgs::msg::Image const & img, int width, int height)
{
  auto cv_img = cv_bridge::toCvCopy(img);
  cv::resize(cv_img->image, cv_img->image, cv::Size(width, height));
  auto img_out = std::make_unique<sensor_msgs::msg::Image>();
  cv_img->toImageMsg(*img_out);
  return img_out;
}

}  // namespace tflite
