import tensorflow as tf

from networks.loss_functions import compute_l1_loss, compute_perceptual_loss, compute_gan_loss

import os
import logging
import unittest
import builtins

using_notebook = getattr(builtins, "__IPYTHON__", False)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

test_batch_size = 4

class TestLossFunction(unittest.TestCase):
  def test_l1_loss(self):
    rand_input = tf.random.normal((test_batch_size, 256, 256, 3))
    same_out = compute_l1_loss(rand_input, rand_input)
    diff_out = compute_l1_loss(rand_input, rand_input - 1)

    self.assertEqual(same_out, 0)
    self.assertNotEqual(diff_out, 0)

  def test_gan_loss(self):
    ones = tf.ones((test_batch_size, 256, 256, 1))
    zeros = tf.zeros((test_batch_size, 256, 256, 1))

    gen_loss, disc_loss = compute_gan_loss(ones, zeros, from_logits=False)

    self.assertEqual(disc_loss, 0)
    self.assertTrue(gen_loss > 0)

    gen_loss, disc_loss = compute_gan_loss(ones, ones, from_logits=False)

    self.assertTrue(disc_loss > 0)
    self.assertEqual(gen_loss, 0)

    gen_loss, disc_loss = compute_gan_loss(zeros, ones, from_logits=False)

    self.assertTrue(disc_loss > 0)

if __name__ == "__main__" and not using_notebook:
  unittest.main()

