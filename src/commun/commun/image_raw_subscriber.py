import rclpy
from rclpy.node import Node

import numpy as np

from sensor_msgs.msg import Image


class ImageRawSubscriber(Node):

    def __init__(self):
        super().__init__('image_raw_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.listener_callback,
            1
        )
        self.subscription  # prevent unused variable warning
        self.l = []

    def listener_callback(self, msg):
        now_ns = self.get_clock().now().nanoseconds
        # self.get_logger().info('I heard: "%s"' % msg.header.stamp)

        ts_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        latency_ns = now_ns - ts_ns
        latency_ms = latency_ns / 1e6

        self.l.append(latency_ms)
        if len(self.l) > 100:
            self.l.pop(0)

        self.get_logger().info(f'Latency: {latency_ms:4f} ns ({np.mean(self.l):.6f} ms)')
        # self.get_logger().info(f'Latency: {latency_ms:4f} ms')


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = ImageRawSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
