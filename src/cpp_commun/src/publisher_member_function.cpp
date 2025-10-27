#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
// #include "std_msgs/msg/string.hpp"
#include "semcom_msgs/msg/test.hpp"

using namespace std::chrono_literals;

/* This example creates a subclass of Node and uses std::bind() to register a
* member function as a callback from the timer. */

class MinimalPublisher : public rclcpp::Node
{
  public:
    MinimalPublisher()
    : Node("minimal_publisher"), count_(0)
    {
      publisher_ = this->create_publisher<semcom_msgs::msg::Test>("topic", 10);
      timer_ = this->create_wall_timer(
        500ms, std::bind(&MinimalPublisher::timer_callback, this)
      );
    }

  private:

    void timer_callback() {
      auto message = semcom_msgs::msg::Test();
      message.first_name = "Leonard";
      message.last_name = "Ngo";
      RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.first_name.c_str());
      publisher_->publish(message);
    }

    // declare variables
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<semcom_msgs::msg::Test>::SharedPtr publisher_;
    size_t count_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalPublisher>());
  rclcpp::shutdown();
  return 0;
}