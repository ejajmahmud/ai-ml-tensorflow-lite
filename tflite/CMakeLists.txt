cmake_minimum_required(VERSION 3.5)
project(tflite)

if(DEFINED ENV{TENSORFLOW_SOURCE_DIR})
  set(TENSORFLOW_SOURCE_DIR $ENV{TENSORFLOW_SOURCE_DIR})
  if(DEFINED ENV{TENSORFLOW_LITE_LIB_DIR})
    set(TENSORFLOW_LITE_LIB_DIR $ENV{TENSORFLOW_LITE_LIB_DIR})
  else()
    set(TENSORFLOW_LITE_LIB_DIR "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/tools/make/gen/linux_x86_64/lib")
  endif()
else()
  # No external TF location given, use the included submodule
  message(NOTICE "TENSORFLOW_SOURCE_DIR not defined, will build our TensorFlow Lite")
  set(TENSORFLOW_SOURCE_DIR ${CMAKE_SOURCE_DIR}/opt/tensorflow)
  set(TENSORFLOW_LITE_GEN_DIR ${CMAKE_INSTALL_PREFIX}/opt/tensorflow/gen/)
  # TODO: set target ourselves or determine directory dynamically
  set(TENSORFLOW_LITE_LIB_DIR "${TENSORFLOW_LITE_GEN_DIR}lib")

  add_custom_command(
    OUTPUT ${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/tools/make/downloads
    COMMAND tensorflow/lite/tools/make/download_dependencies.sh
    WORKING_DIRECTORY ${TENSORFLOW_SOURCE_DIR})

  add_custom_command(
    OUTPUT ${TENSORFLOW_LITE_LIB_DIR}/libtensorflow-lite.a
    COMMAND make GENDIR=${TENSORFLOW_LITE_GEN_DIR} -j $(nproc) -f tensorflow/lite/tools/make/Makefile
    DEPENDS ${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/tools/make/downloads
    WORKING_DIRECTORY ${TENSORFLOW_SOURCE_DIR})

  add_custom_target(
    libtensorflow_lite ALL
    DEPENDS ${TENSORFLOW_LITE_LIB_DIR}/libtensorflow-lite.a)
endif()

message("TensorFlow source directory: ${TENSORFLOW_SOURCE_DIR}")
message("TensorFlow Lite lib directory: ${TENSORFLOW_LITE_LIB_DIR}")

set(tensorflow_lite_INCLUDE_DIRS
  ${TENSORFLOW_SOURCE_DIR}
  ${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/tools/make/downloads/flatbuffers/include)
set(tensorflow_lite_LIBRARIES tensorflow-lite dl)

link_directories(${TENSORFLOW_LITE_LIB_DIR})

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)

add_library(tflite SHARED
  src/visionnode.cpp
  src/classification.cpp
  src/objectdetection.cpp
  src/semanticsegmentation.cpp)

target_include_directories(tflite PUBLIC
  ${tensorflow_lite_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

ament_target_dependencies(tflite
  "rclcpp"
  "cv_bridge"
  "rclcpp_components"
  "sensor_msgs"
  "vision_msgs"
)

# If we buikd TFLite ourselves, add it as dependency
if(TARGET libtensorflow_lite)
  add_dependencies(tflite libtensorflow_lite)
endif()

target_link_libraries(tflite tensorflow-lite)

# check for libedgetpu
find_path(edgetpu_INCLUDE_DIR NAMES edgetpu_c.h)
find_library(edgetpu_LIBRARY NAMES edgetpu)
if(edgetpu_INCLUDE_DIR AND edgetpu_LIBRARY)
  target_compile_definitions(tflite PRIVATE HAVE_EDGETPU)
  target_include_directories(tflite PRIVATE ${edgetpu_INCLUDE_DIR})
  target_link_libraries(tflite ${edgetpu_LIBRARY})
  message("Edge TPU include dir: ${edgetpu_INCLUDE_DIR}")
  message("Edge TPU library: ${edgetpu_LIBRARY}")
endif()

rclcpp_components_register_nodes(tflite
  "tflite::SemanticSegmentation"
  "tflite::Classification"
  "tflite::ObjectDetection"
)

# target_compile_options(tflite PRIVATE -Werror)

# Causes the visibility macros to use dllexport rather than dllimport,
# which is appropriate when building the dll but not consuming it.
target_compile_definitions(tflite PRIVATE "TFLITE_BUILDING_LIBRARY")
target_link_libraries(tflite rt)

install(
  DIRECTORY include/
  DESTINATION include
)
install(
  TARGETS tflite
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Semantic segmentation node
add_executable(semanticsegmentation_node
  src/semanticsegmentation_node.cpp)
target_include_directories(semanticsegmentation_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(semanticsegmentation_node tflite)
# target_compile_options(tflite_node PRIVATE -Werror)

install(TARGETS semanticsegmentation_node
  EXPORT export_${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME})

# Classification node
add_executable(classification_node
  src/classification_node.cpp)
target_include_directories(classification_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(classification_node tflite)
# target_compile_options(tflite_node PRIVATE -Werror)

install(TARGETS classification_node
  EXPORT export_${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME})

# Object detection node
add_executable(objectdetection_node
  src/objectdetection_node.cpp)
target_include_directories(objectdetection_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(objectdetection_node tflite)
# target_compile_options(tflite_node PRIVATE -Werror)

install(TARGETS objectdetection_node
  EXPORT export_${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  set(ament_cmake_pep257_FOUND TRUE)
  set(ament_cmake_flake8_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
