import rclpy
from rclpy.node import Node

import numpy as np

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from semcom_msgs.msg import Code


class CodePublisher(Node):

    def __init__(self):
        super().__init__('code_publisher')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.listener_callback,
            1
        )
        self.publisher_ = self.create_publisher(
            Code,
            '/camera/camera/color/code',
            1
        )

    def listener_callback(self, msg):
        # self.get_logger().info('I heard: "%s"' % msg.header.stamp)
        
        msg = Code()
        header = Header()
        header.frame_id = "semcom"
        header.stamp = self.get_clock().now().to_msg()
        msg.header = header
        msg.data = np.random.randint(0, 8192, (150,), dtype=np.int16)
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing:')


def main(args=None):
    rclpy.init(args=args)

    publisher = CodePublisher()

    rclpy.spin(publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
