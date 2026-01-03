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

from taming.models.vqgan import VQModel

from std_msgs.msg import Header
from semcom_msgs.msg import Code

from skimage import data
from skimage.transform import resize


class ImageCodeSubscriber(Node):

	def __init__(self):
		super().__init__('image_code_subscriber')
		self.subscription = self.create_subscription(
			CompressedImage,
			'/camera/camera/color/image_raw/compressed',
			self.listener_callback,
			1
		)
		self.publisher = self.create_publisher(
			Image,
			'/decode',
			1
		)
		self.code_publisher = self.create_publisher(
			Code, '/camera/camera/color/image_raw/code',
			1
		)

		config = get_config_from_file("./src/commun/commun/imagenet_vitvq_base.yaml")
		self.model: ViTVQ = initialize_from_config(config.model)
		self.model.init_from_ckpt("./src/commun/commun/checkpoint/imagenet_vitvq_base.ckpt")
		for param in self.model.parameters():
			param.requires_grad = False
		self.model.eval()
		self.model.cuda()

		# config = get_config_from_file("./src/commun/commun/vqgan_f16_16384.yaml")  # 2^14
		# self.model: VQModel = initialize_from_config(config.model)
		# self.model.init_from_ckpt("./src/commun/commun/checkpoint/vqgan_f16_16384.ckpt")
		# for param in self.model.parameters():
		# 	param.requires_grad = False
		# self.model.eval()
		# self.model.cuda()
		
		# self.quantizer = vitvq.quantizer
		# self.post_quant = vitvq.post_quant
		# self.decoder = vitvq.decoder
		# del vitvq

		self.bits_per_code = 14
		self.l = []
		self.mse = []
		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.bridge = CvBridge()
		self.base_size = (256, 256)
		self.ratio = (1, 1)

	def encode_codes(self, x: torch.FloatTensor) -> torch.LongTensor:
		h = self.encoder(x)
		h = self.pre_quant(h)
		_, _, codes = self.quantizer(h)
		
		return codes
	
	def publish_code(self, codes, frame_id):

		codes = codes.detach().to("cpu").contiguous().numpy()
		# self.get_logger().info(f'codes: {"-".join([str(c) for c in codes[0, :10]])}')

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
		_msg.header.frame_id = frame_id

		_msg.length = 1024*self.ratio[0]*self.ratio[1]
		_msg.data = packed.tolist()  # uint8[]

		self.code_publisher.publish(_msg)

	def listener_callback(self, msg: CompressedImage):
		now_ns = self.get_clock().now().nanoseconds
		# self.get_logger().info('I heard: "%s"' % msg.header.stamp)

		ts_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
		latency_ns = now_ns - ts_ns
		latency_ms = latency_ns / 1e6

		self.l.append(latency_ms)
		if len(self.l) > 100: self.l.pop(0)
		
		# parsing incoming msg
		np_arr = np.frombuffer(msg.data, dtype=np.uint8)
		img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		
		image_resized = resize(
			img, (self.base_size[0]*self.ratio[0], self.base_size[1]*self.ratio[1]),
			preserve_range=True, anti_aliasing=True
		).astype(np.uint8)
		input_data = (
			torch.from_numpy(image_resized)
			.permute(2, 0, 1)
			.unsqueeze(0)
			.to(self.device)
			/ 255.0
		)

		batch_input_data = input_data.reshape(1, 3, self.ratio[0], self.base_size[0], self.ratio[1], self.base_size[1])      	# [1, 3, H_blocks, 256, W_blocks, 256]
		batch_input_data = batch_input_data.permute(0, 2, 4, 1, 3, 5)		# [1, 3, 2, 3, 256, 256]
		batch_input_data = batch_input_data.reshape(self.ratio[0]*self.ratio[1], 3, self.base_size[0], self.base_size[1])         # [6, 3, 256, 256]
		# print(input_data.shape, batch_input_data.shape)

		with torch.no_grad():
			codes = self.model.encode_codes(batch_input_data)
			dec = self.model.decode_codes(codes)

			# _, _, _codes = self.model.encode(batch_input_data*2 -1)
			# codes = _codes[-1]
			# z_q = self.model.quantize.embedding(codes).reshape(self.ratio[0]*self.ratio[1], 16, 16, 256).permute(0, 3, 1, 2)
			# dec = self.model.decode(z_q).add(1).div(2)#.cpu().squeeze().permute(1, 2, 0)
		self.publish_code(codes, msg.header.frame_id)

		_dec = dec.reshape(1, self.ratio[0], self.ratio[1], 3, self.base_size[0], self.base_size[1])   # [1, H_blocks, W_blocks, C, 256, 256]
		_dec = _dec.permute(0, 3, 1, 4, 2, 5)       # [1, 3, 3, 256, 2, 256]
		_dec = _dec.reshape(1, 3, self.base_size[0]*self.ratio[0], self.base_size[1]*self.ratio[1])         # [1, 3, 768, 512]
		dec_img = _dec.squeeze(0).cpu().permute(1, 2, 0).numpy()
		dec_img = dec_img.clip(0, 1)
		# dec_img = (dec_img)

		dec_img = (dec_img * 255).astype(np.uint8)
		# plt.imshow(dec_img)
		# plt.savefig("./img.png")
		# resize: (width=424, height=240)
		dec_img = cv2.resize(
			dec_img, (img.shape[1], img.shape[0]),
			interpolation=cv2.INTER_LINEAR
		)
		mse = np.mean((img/255. - dec_img/255.) ** 2)
		self.mse.append(mse)
		if len(self.mse) > 100: self.mse.pop(0)
		self.get_logger().info(f'Latency: {latency_ms:4f} ms ({np.mean(self.l):.6f} ms), avg mse: {np.mean(self.mse)}')

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
