import rclpy
from rclpy.node import Node

import numpy as np
import cv2

from skimage.transform import resize

import torch
from torch import nn

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
# from .vitvq-gan import ViTVQ
from enhancing.modules.stage1.vitvqgan import ViTVQ
from enhancing.modules.stage1.layers import ViTEncoder as Encoder, ViTDecoder as Decoder
from enhancing.modules.stage1.quantizers import VectorQuantizer, GumbelQuantizer
from enhancing.utils.general import get_config_from_file, initialize_from_config

from std_msgs.msg import Header
from semcom_msgs.msg import Code


class ImageCodeSubscriber(Node):

	def __init__(self):
		super().__init__('image_code_subscriber')
		self.subscription = self.create_subscription(
			Code,
			'/camera/camera/color/image_raw/code',
			self.listener_callback,
			10
		)
		self.publisher = self.create_publisher(
			Image,
			'/decode',
			1
		)

		config = get_config_from_file("./src/commun/commun/imagenet_vitvq_small.yaml")
		vitvq: ViTVQ = initialize_from_config(config.model)
		vitvq.init_from_ckpt("./src/commun/commun/checkpoint/imagenet_vitvq_small.ckpt")
		for param in vitvq.parameters():
			param.requires_grad = False
		vitvq.eval()
		vitvq.cuda()
		
		self.quantizer = vitvq.quantizer
		self.post_quant = vitvq.post_quant
		self.decoder = vitvq.decoder
		# del vitvq

		self.bits_per_code = 13
		self.l = []

		self.bridge = CvBridge()
		self.base_size = (256, 256)
		self.ratio = (2, 3)

	def encode_codes(self, x: torch.FloatTensor) -> torch.LongTensor:
		h = self.encoder(x)
		h = self.pre_quant(h)
		_, _, codes = self.quantizer(h)
		
		return codes
	
	def listener_callback(self, msg: Code):
		now_ns = self.get_clock().now().nanoseconds
		# self.get_logger().info('I heard: "%s"' % msg.header.stamp)

		ts_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
		latency_ns = now_ns - ts_ns
		latency_ms = latency_ns / 1e6

		self.l.append(latency_ms)
		if len(self.l) > 100: self.l.pop(0)

		self.get_logger().info(f'Latency: {latency_ms:4f} ms ({np.mean(self.l):.6f} ms)')
		
		# parsing
		byte_arr = np.frombuffer(msg.data, dtype=np.uint8)
		bitstream = np.unpackbits(byte_arr, bitorder="big")
		bitstream = bitstream[:msg.length*self.bits_per_code]
		bits = bitstream.reshape(msg.length, self.bits_per_code)
		codes_np = (bits << np.arange(12, -1, -1)).sum(axis=1).astype(np.uint16)

		if codes_np.size != msg.length:
			self.get_logger().error(f"Code length mismatch: {codes_np.shape}-{msg.length}")
			return

		self.get_logger().info(f'codes: {"-".join([str(c) for c in codes_np[:10]])}')
		codes = torch.from_numpy(codes_np).long().reshape(-1, 1024).cuda()

		# decoding
		with torch.no_grad():
			quant = self.quantizer.embedding(codes.cuda())
			quant = self.quantizer.norm(quant)
			
			if self.quantizer.use_residual:
				quant = quant.sum(-2)  
				
			quant = self.post_quant(quant)
			dec = self.decoder(quant)
		
		_dec = dec.reshape(1, self.ratio[0], self.ratio[1], 3, self.base_size[0], self.base_size[1])   # [1, H_blocks, W_blocks, C, 256, 256]
		_dec = _dec.permute(0, 3, 1, 4, 2, 5)       # [1, 3, 3, 256, 2, 256]
		_dec = _dec.reshape(1, 3, self.base_size[0]*self.ratio[0], self.base_size[1]*self.ratio[1])         # [1, 3, 768, 512]
		
		dec_img = _dec.squeeze(0).permute(1, 2, 0).cpu().numpy()
		dec_img = dec_img.clip(0, 1)

		dec_img = (dec_img * 255).astype(np.uint8)
		# plt.imshow(dec_img)
		# plt.savefig("./img.png")
		# resize: (width=424, height=240)
		dec_img = cv2.resize(
			dec_img, (640, 480),
			interpolation=cv2.INTER_LINEAR
		)
		_msg = self.bridge.cv2_to_imgmsg(dec_img, encoding="rgb8")
		_msg.header.stamp = self.get_clock().now().to_msg()
		_msg.header.frame_id = msg.header.frame_id

		self.publisher.publish(_msg)
		

def main(args=None):
	rclpy.init(args=args)

	subscriber = ImageCodeSubscriber()

	rclpy.spin(subscriber)

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	subscriber.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
