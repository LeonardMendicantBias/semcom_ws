cmake_minimum_required(VERSION 3.8)
project(cpp_commun)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(semcom_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# TensorRT libraries
find_library(NVINFER_LIB nvinfer REQUIRED)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin REQUIRED)

# CUDA
find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h
          HINTS /usr/local/cuda/include)
find_library(CUDA_LIB cuda
             HINTS /usr/local/cuda/lib64)
find_library(CUDART_LIB cudart
  HINTS
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    /usr/local/cuda/lib64
)
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIR}      # <<< important
)


add_executable(talker src/publisher_member_function.cpp)
ament_target_dependencies(talker rclcpp std_msgs semcom_msgs)

add_executable(listener src/subscriber_member_function.cpp)
ament_target_dependencies(listener rclcpp std_msgs semcom_msgs)

add_executable(sem_publisher src/sem_publisher.cpp)
ament_target_dependencies(sem_publisher rclcpp std_msgs sensor_msgs semcom_msgs cv_bridge OpenCV)
# Target-specific include dirs and libs (safer than global include_directories/link)
target_include_directories(sem_publisher PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIR}
)
target_link_libraries(sem_publisher
  ${OpenCV_LIBS}
  ${NVINFER_LIB}
  ${NVINFER_PLUGIN_LIB}
  ${CUDA_LIB}
  ${CUDART_LIB} 
)

add_executable(sem_subscriber src/sem_subscriber.cpp)
ament_target_dependencies(sem_subscriber rclcpp std_msgs sensor_msgs semcom_msgs OpenCV)


install(TARGETS
  talker
  listener
  sem_publisher
  sem_subscriber
  DESTINATION lib/${PROJECT_NAME})
install(
  DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
)
# rosidl_get_typesupport_target(cpp_typesupport_target
#   ${PROJECT_NAME} rosidl_typesupport_cpp)

# target_link_libraries(publisher_member_function "${cpp_typesupport_target}")

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
