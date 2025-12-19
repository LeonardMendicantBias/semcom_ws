import rclpy
from rclpy.node import Node

import numpy as np
import cv2

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
		self.subscription = self.create_subscription(
			CompressedImage,
			'/camera/camera/color/image_raw/compressed',
			self.listener_callback,
			1
		)
		self.publisher = self.create_publisher(
			Code, '/camera/camera/color/image_raw/code',
			1
		)

		config = get_config_from_file("/home/leonard/semcom_ws/src/commun/commun/imagenet_vitvq_small.yaml")
		vitvq: ViTVQ = initialize_from_config(config.model)
		vitvq.init_from_ckpt("/home/leonard/semcom_ws/src/commun/commun/checkpoint/imagenet_vitvq_small.ckpt")
		for param in vitvq.parameters():
			param.requires_grad = False
		vitvq.eval()
		vitvq.cuda()
		
		self.encoder = vitvq.encoder
		self.quantizer = vitvq.quantizer
		self.pre_quant = vitvq.pre_quant
		del vitvq

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.l = []

		self.bits_per_code = 13

	def encode_codes(self, x: torch.FloatTensor) -> torch.LongTensor:
		h = self.encoder(x)
		h = self.pre_quant(h)
		_, _, codes = self.quantizer(h)
		
		return codes
	
	def listener_callback(self, msg: CompressedImage):
		now_ns = self.get_clock().now().nanoseconds
		# self.get_logger().info('I heard: "%s"' % msg.header.stamp)

		ts_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
		latency_ns = now_ns - ts_ns
		latency_ms = latency_ns / 1e6

		self.l.append(latency_ms)
		if len(self.l) > 100: self.l.pop(0)

		self.get_logger().info(f'Latency: {latency_ms:4f} ms ({np.mean(self.l):.6f} ms)')
		# self.get_logger().info(f'Latency: {latency_ms:4f} ms')


		np_arr = np.frombuffer(msg.data, dtype=np.uint8)
		image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # BGR

		image_resized = resize(image, (256, 256), anti_aliasing=True)
		input_data = (
			torch.from_numpy(image_resized)
			.permute(2, 0, 1)
			.unsqueeze(0)
			.to(self.device, non_blocking=True)
			.float()
			/ 255.0
		)
		# with torch.no_grad():
		# encoding
		with torch.no_grad(): #, torch.autocast("cuda", torch.float16):
			h = self.encoder(input_data)
			h = self.pre_quant(h)
			_, _, codes = self.quantizer(h)
		self.get_logger().info(f'codes: {codes.shape}')

		codes = codes.detach().to("cpu").contiguous()
		codes_np = codes.view(-1).numpy().astype(np.uint16)

		# pack bits into bytes
		packed = np.packbits(
			np.unpackbits(codes_np[:, None].view(np.uint8), axis=1)[:, -self.bits_per_code:],
			axis=1
		).reshape(-1)

		msg = Code()
		msg.header = Header()
		msg.header.stamp = self.get_clock().now().to_msg()
		msg.header.frame_id = "codes"

		msg.length = 1024
		msg.data = packed.tolist()  # uint8[]

		self.publisher.publish(msg)


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
