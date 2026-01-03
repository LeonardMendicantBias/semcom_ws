import time

import rclpy
from rclpy.node import Node

import numpy as np

from semcom_msgs.msg import Code


class CodeSubscriber(Node):

	def __init__(self):
		super().__init__('code_subscriber')
		self.subscription = self.create_subscription(
			Code,
			'/camera/camera/color/code',
			self.listener_callback,
			1
		)
		self.subscription  # prevent unused variable warning
		self.l = []
		self.count = 0
		self.latencies = []

	def listener_callback(self, msg):
		t_recv_ns = time.time_ns()

		t_send_ns = (
			msg.header.stamp.sec * 1_000_000_000 + 
			msg.header.stamp.nanosec
		)
		latency_ms = (t_recv_ns - t_send_ns) / 1e6
		self.latencies.append(latency_ms)

		self.l.append(latency_ms)
		if len(self.l) > 100:
			self.l.pop(0)

		data = np.array(msg.data, dtype=np.int16)
		# self.get_logger().info(f'{len(data)} Latency: {latency_ms:4f} ns ({np.mean(self.l):.6f} ms)')
		# self.get_logger().info(f'Latency: {latency_ms:4f} ms')

		self.count += 1
		self.get_logger().info(f'Received {self.count} {len(data)} images with latency {latency_ms:.4f}-{np.mean(self.latencies):.4f}/{np.std(self.latencies):.4f}')


def main(args=None):
	rclpy.init(args=args)

	subscriber = CodeSubscriber()

	rclpy.spin(subscriber)

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	subscriber.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
