#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <cstdlib>

#include "rclcpp/rclcpp.hpp"
// #include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "semcom_msgs/msg/code.hpp"

using namespace std::chrono_literals;

class SemPublisher : public rclcpp::Node
{
	public:
		SemPublisher() : Node("Semantic_Node")
		{
			// subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
			// "/camera/camera/color/image_raw", 10, std::bind(&SemPublisher::topic_callback, this, _1));

			publisher_ = this->create_publisher<semcom_msgs::msg::Code>(
			"/camera/camera/color/image_code", 10);
      timer_ = this->create_wall_timer(
        500ms, std::bind(&SemPublisher::timer_callback, this)
      );
		}

	private:
		void timer_callback() const
		{
			const int N = 250;
			const int BITS_PER_VALUE = 13;
			const int TOTAL_BITS = N * BITS_PER_VALUE;

			auto message = semcom_msgs::msg::Code();
			message.header.stamp = this->now();
			message.header.frame_id = "sem_com";

			std::vector<uint8_t> bitstream;
			// reserve memory to the least bytes
			bitstream.reserve((TOTAL_BITS + 7) / 8);

			uint8_t current_byte = 0;
			int bits_in_current = 0;
			for (int i=0; i<N; i++) {
				int randomNumber = rand() % 8192;

				// pack 13 bits (MSB-first)
				for (int b = BITS_PER_VALUE - 1; b >= 0; --b) {
						uint8_t bit = (randomNumber >> b) & 1;
						current_byte = (current_byte << 1) | bit;
						bits_in_current++;

						if (bits_in_current == 8) {
								bitstream.push_back(current_byte);
								current_byte = 0;
								bits_in_current = 0;
						}
				}
				
			}
			if (bits_in_current > 0) {
				// pad remaining bits in the last byte
				current_byte <<= (8 - bits_in_current); // shift bits to MSB positions
				bitstream.push_back(current_byte);
			}

			
			RCLCPP_INFO(this->get_logger(), "Publishing");
			
			message.length = N;
			message.data = bitstream;
			publisher_->publish(message);
		}

		// void topic_callback(const semcom_msgs::msg::Code & msg) const
		// {
		// 	int N = 5;
		// 	auto message = semcom_msgs::msg::Code();
		// 	message.header.stamp = this->now();
		// 	message.header.frame_id = "sem_com";

		// 	// std::vector<uint8_t> bytes;
		// 	// bytes.reserve((N * 13 + 7) / 8);
		// 	// for (int i=0; i<N; i++) {
		// 	// 	int randomNumber = rand() % 8192;
		// 	// 	uint8_t bit = (v >> b) & 0x1;
		// 	// }
			
		// 	RCLCPP_INFO(this->get_logger(), "Publishing");
		// }

		// rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
		rclcpp::Publisher<semcom_msgs::msg::Code>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<SemPublisher>());
	rclcpp::shutdown();
	return 0;
}
