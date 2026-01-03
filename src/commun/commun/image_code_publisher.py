import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

import numpy as np
import cv2

from cv_bridge import CvBridge

from skimage.transform import resize

import torch
from torch import nn

from sensor_msgs.msg import Image, CompressedImage
# from .vitvq-gan import ViTVQ
from enhancing.modules.stage1.vitvqgan import ViTVQ
from enhancing.modules.stage1.layers import ViTEncoder as Encoder, ViTDecoder as Decoder
from enhancing.modules.stage1.quantizers import VectorQuantizer, GumbelQuantizer
from enhancing.utils.general import get_config_from_file, initialize_from_config

from std_msgs.msg import Header
from semcom_msgs.msg import Code


class ImageCodePublisher(Node):

	def __init__(self):
		super().__init__('image_code_publisher')

		self.bridge = CvBridge()
		self.subscription = self.create_subscription(
			# CompressedImage,
			Image,
			'/camera/camera/color/image_raw',
			self.listener_callback,
			qos_profile=QoSProfile(
				history=HistoryPolicy.KEEP_LAST,
				depth=1,
				reliability=ReliabilityPolicy.BEST_EFFORT,
				# durability=DurabilityPolicy.VOLATILE
			)
		)
		self.publisher = self.create_publisher(
			Code, '/camera/camera/color/image_raw/code',
			1
		)

		config = get_config_from_file("./src/commun/commun/imagenet_vitvq_encoder_small.yaml")
		self.vitvq: ViTVQ = initialize_from_config(config.model)
		for param in self.vitvq.parameters():
			param.requires_grad = False
		self.vitvq.eval()
		self.vitvq.cuda()

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.l = []

		self.bits_per_code = 13
		self.base_size = (256, 256)
		self.ratio = (1, 2)
		
	def listener_callback(self, msg: Image):
		now_ns = self.get_clock().now().nanoseconds

		ts_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
		latency_ns = now_ns - ts_ns
		latency_ms = latency_ns / 1e6

		self.l.append(latency_ms)
		if len(self.l) > 100: self.l.pop(0)

		self.get_logger().info(f'Latency: {latency_ms:4f} ms ({np.mean(self.l):.6f} ms)')

		img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
		
		image_resized = resize(
			img, (self.base_size[0]*self.ratio[0], self.base_size[1]*self.ratio[1]),
			preserve_range=True, anti_aliasing=True
		).astype(np.uint8)
		input_data = (
			torch.from_numpy(image_resized)
			.permute(2, 0, 1)
			.unsqueeze(0)
			.to(self.device)
			.float()
			/ 255.0
		)

		batch_input_data = input_data.reshape(1, 3, self.ratio[0], self.base_size[0], self.ratio[1], self.base_size[1])      	# [1, 3, H_blocks, 256, W_blocks, 256]
		batch_input_data = batch_input_data.permute(0, 2, 4, 1, 3, 5)		# [1, 3, 2, 3, 256, 256]
		batch_input_data = batch_input_data.reshape(self.ratio[0]*self.ratio[1], 3, self.base_size[0], self.base_size[1])         # [6, 3, 256, 256]
		
		# encoding
		with torch.no_grad(), torch.autocast("cuda", torch.float16):
			codes = self.vitvq.encode_codes(batch_input_data)

		codes = codes.detach().to("cpu").contiguous().numpy()
		print(codes.shape)
		self.get_logger().info(f'codes: {"-".join([str(c) for c in codes[0, :10]])}')

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
		_msg.header.stamp = self.get_clock().now().to_msg()
		_msg.header.frame_id = msg.header.frame_id

		_msg.length = 1024*self.ratio[0]*self.ratio[1]
		_msg.data = packed.tolist()  # uint8[]

		self.publisher.publish(_msg)


def main(args=None):
	rclpy.init(args=args)

	subscriber = ImageCodePublisher()

	rclpy.spin(subscriber)

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	subscriber.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()



		# config = get_config_from_file("/home/nvidia/semcom_ws/src/commun/commun/imagenet_vitvq_encoder_base.yaml")

		# self.vitvq_base: ViTVQ = initialize_from_config(config.model)
		# for param in self.vitvq.parameters():
		# 	param.requires_grad = False
		# self.vitvq_base.eval()
		# self.vitvq_base.cuda()