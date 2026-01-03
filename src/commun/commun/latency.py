import rclpy
from rclpy.node import Node

import numpy as np

from sensor_msgs.msg import Image, CompressedImage


class Latency(Node):

    def __init__(self):
        super().__init__('latency')
        self.subscription = self.create_subscription(
            CompressedImage,
            # '/camera/camera/color/image_raw/compressed',
            '/image_q60/compressed',
            self.callback,
            1
        )
        self.subscription  # prevent unused variable warning
        self.l = []

    def callback(self, msg):
        now_ns = self.get_clock().now().nanoseconds

        ts_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        latency_ns = now_ns - ts_ns
        latency_ms = latency_ns / 1e6

        self.l.append(latency_ms)
        if len(self.l) > 100:
            self.l.pop(0)

        self.get_logger().info(f'Latency: {latency_ms:4f} ms ({np.mean(self.l):.6f} ms)')
        # self.get_logger().info(f'Latency: {latency_ms:4f} ms')


def main(args=None):
    rclpy.init(args=args)

    subscriber = Latency()

    rclpy.spin(subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
