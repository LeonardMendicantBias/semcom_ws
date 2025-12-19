import rclpy
from rclpy.node import Node

import numpy as np
import cv2

from sensor_msgs.msg import Image, CompressedImage
# from .vitvq-gan import ViTVQ
# from enhancing import ViTVQ
# from enhancing import ViTEncoder as Encoder, ViTDecoder as Decoder
# from enhancing import VectorQuantizer, GumbelQuantizer


class ImageCodeSubscriber(Node):

    def __init__(self):
        super().__init__('image_raw_subscriber')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/camera/color/image_raw/compressed',
            self.listener_callback,
            1
        )
        self.subscription  # prevent unused variable warning
        self.l = []

    def listener_callback(self, msg: CompressedImage):
        pass


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = ImageCodeSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
