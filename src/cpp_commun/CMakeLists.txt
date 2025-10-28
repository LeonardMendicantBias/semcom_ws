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

add_executable(talker src/publisher_member_function.cpp)
ament_target_dependencies(talker rclcpp std_msgs semcom_msgs)

add_executable(listener src/subscriber_member_function.cpp)
ament_target_dependencies(listener rclcpp std_msgs semcom_msgs)

add_executable(sem_publisher src/sem_publisher.cpp)
ament_target_dependencies(sem_publisher rclcpp std_msgs sensor_msgs semcom_msgs)

add_executable(sem_subscriber src/sem_subscriber.cpp)
ament_target_dependencies(sem_subscriber rclcpp std_msgs sensor_msgs semcom_msgs)

install(TARGETS
  talker
  listener
  sem_publisher
  sem_subscriber
  DESTINATION lib/${PROJECT_NAME})

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
