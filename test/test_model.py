import tensorflow as tf
from networks.models import RRDBNet, UNetDiscriminator, Vgg19FeaturesModel

import os
import logging

import unittest
import builtins

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

using_notebook = getattr(builtins, "__IPYTHON__", False)

test_batch_size = 4

class TestModels(unittest.TestCase):
  def setUp(self):
    self.generator_test = RRDBNet()
    self.discriminator_test = UNetDiscriminator()
    self.vgg_test = Vgg19FeaturesModel()

    self.generator_input = tf.random.normal((test_batch_size, 64, 64, 3))
    self.discriminator_input = tf.random.normal((test_batch_size, 256, 256, 3))

    self.generator_output_shape = (test_batch_size, 256, 256, 3)
    self.discriminator_output_shape = (test_batch_size, 256, 256, 1)

  def test_generator(self):
    gen_out = self.generator_test(self.generator_input)
    self.assertEqual(gen_out.shape, self.generator_output_shape)

  def test_discriminator(self):
    disc_out = self.discriminator_test(self.discriminator_input)
    self.assertEqual(disc_out.shape, self.discriminator_output_shape)

  def test_vgg(self):
    vgg_out = self.vgg_test(self.discriminator_input)

    self.assertEqual(vgg_out[0].shape, (test_batch_size, 256, 256, 64))
    self.assertEqual(vgg_out[1].shape, (test_batch_size, 128, 128, 128))
    self.assertEqual(vgg_out[2].shape, (test_batch_size, 64, 64, 256))
    self.assertEqual(vgg_out[3].shape, (test_batch_size, 32, 32, 512))
    self.assertEqual(vgg_out[4].shape, (test_batch_size, 16, 16, 512))

if __name__ == "__main__" and not using_notebook:
  unittest.main()

