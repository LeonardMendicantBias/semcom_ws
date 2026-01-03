import time

import rclpy
from rclpy.node import Node

import numpy as np

from std_msgs.msg import Header
from sensor_msgs.msg import Image, CompressedImage
from semcom_msgs.msg import Code
import torch


class CodePublisher(Node):

	def __init__(self):
		super().__init__('code_publisher')
		
		
		self.sub = self.create_subscription(
			CompressedImage,
			'/camera/camera/color/image_raw/compressed',
			# '/camera/camera/color/image_raw/compressed',
			self.callback,
			1
		)
		self.publisher_ = self.create_publisher(
			Code,
			'/camera/camera/color/code',
			1
		)

		self.ratio = (2, 2)
		self.count = 0
		self.max_count = 1000

	def callback(self, msg):
		if self.count > self.max_count: 
			self.get_logger().info('Reached 1000 images. Shutting down node.')
			rclpy.shutdown()
			return
		self.get_logger().info(f'Published {self.count} codes.')

		codes = torch.randint(0, 2**13, (self.ratio[0]*self.ratio[1], 1024)).numpy()

		codes = codes.astype(np.uint16).reshape(-1)
		bits = ((codes[:, None] >> np.arange(12, -1, -1)) & 1).astype(np.uint8)

		# flatten bitstream
		bitstream = bits.reshape(-1)

		# pad to byte boundary
		pad = (-len(bitstream)) % 8
		if pad:
			bitstream = np.pad(bitstream, (0, pad), constant_values=0)

		# bits → bytes
		packed = np.packbits(bitstream, bitorder="big")

		_msg = Code()
		_msg.header = Header()
		_msg.header.frame_id = msg.header.frame_id
		# _msg.header.stamp = self.get_clock().now().to_msg()
		t_send_ns = time.time_ns()
		_msg.header.stamp.sec = t_send_ns // 1_000_000_000
		_msg.header.stamp.nanosec = t_send_ns % 1_000_000_000

		_msg.length = 1024*self.ratio[0]*self.ratio[1]
		_msg.data = packed.tolist()  # uint8[]

		self.publisher_.publish(_msg)
		self.count += 1


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
