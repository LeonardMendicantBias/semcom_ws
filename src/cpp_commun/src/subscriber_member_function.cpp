#include <memory>

#include "rclcpp/rclcpp.hpp"
// #include "std_msgs/msg/string.hpp"
#include "semcom_msgs/msg/test.hpp"

using std::placeholders::_1;

class MinimalSubscriber : public rclcpp::Node
{
  public:
    MinimalSubscriber():Node("minimal_subscriber")
    {
      subscription_ = this->create_subscription<semcom_msgs::msg::Test>(
      "topic", 10, std::bind(&MinimalSubscriber::topic_callback, this, _1));
    }

  private:
    void topic_callback(const semcom_msgs::msg::Test & msg) const
    {
      RCLCPP_INFO(this->get_logger(), "I heard: '%s' '%s'", msg.first_name.c_str(), msg.last_name.c_str());
    }

    rclcpp::Subscription<semcom_msgs::msg::Test>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalSubscriber>());
  rclcpp::shutdown();
  return 0;
}
