#include <memory>

#include "rclcpp/rclcpp.hpp"
// #include "std_msgs/msg/string.hpp"
#include "semcom_msgs/msg/code.hpp"

using std::placeholders::_1;

class SemSubscriber : public rclcpp::Node
{
  public:
    SemSubscriber():Node("Sem_Subscriber")
    {
      subscription_ = this->create_subscription<semcom_msgs::msg::Code>(
      "/camera/camera/color/image_code", 10, std::bind(&SemSubscriber::topic_callback, this, _1));
    }

  private:
    void topic_callback(const semcom_msgs::msg::Code::SharedPtr msg)
    {
      const uint16_t count = msg->length;
      const size_t BITS_PER_VALUE = 13;
      const size_t total_bits = static_cast<size_t>(count) * BITS_PER_VALUE;
      const size_t required_bytes = (total_bits + 7) / 8;
      
      if (count == 0) {
        RCLCPP_WARN(this->get_logger(), "Received length=0 -> nothing to decode");
        return;
      }
      if (msg->data.size() < required_bytes) {
        RCLCPP_ERROR(this->get_logger(),
                    "Insufficient bytes: have %zu, need %zu for length=%u (bits=%zu)",
                    msg->data.size(), required_bytes, count, total_bits);
        return;
      }

      auto read_bit = [&](size_t bit_index) -> uint8_t {
        size_t byte_idx = bit_index / 8;
        size_t bit_in_byte = bit_index % 8;
        return static_cast<uint8_t>((msg->data[byte_idx] >> (7 - bit_in_byte)) & 0x1);
      };

      std::vector<uint16_t> values;
      values.reserve(count);

      for (size_t i = 0; i < count; ++i) {
        uint16_t val = 0;
        size_t start_bit = i * BITS_PER_VALUE;
        for (size_t b = 0; b < BITS_PER_VALUE; ++b) {
          uint8_t bit = read_bit(start_bit + b);
          val = static_cast<uint16_t>((val << 1) | bit);
        }
        values.push_back(val);
      }

      std::stringstream ss;
      ss << msg->header.stamp.sec << "." << msg->header.stamp.nanosec;
      RCLCPP_INFO(this->get_logger(), "I heard: '%s'", ss.str().c_str());
      RCLCPP_INFO(this->get_logger(),
                  "Decoded %u values (bits=%zu, bytes=%zu). First 10: %s",
                  count, total_bits, msg->data.size(),
                  preview_values(values, 10).c_str());
    }

    std::string preview_values(const std::vector<uint16_t>& v, size_t n) {
      size_t m = std::min(n, v.size());
      std::string s;
      for (size_t i = 0; i < m; ++i) {
        if (i) s += ", ";
        s += std::to_string(v[i]);
      }
      return s;
    }

    rclcpp::Subscription<semcom_msgs::msg::Code>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SemSubscriber>());
  rclcpp::shutdown();
  return 0;
}
