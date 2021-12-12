from data_generator.data_generator import train_dataset, feed_data, PoolData
from data_generator.data_generator import feed_props_1, feed_props_2

import os
import logging
import unittest
import builtins

using_notebook = getattr(builtins, "__IPYTHON__", False)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

test_batch_size = 4
pad_img_size = 400
final_gt_img_size = 256
final_lq_img_size = 64

kernel_size = 21

test_pool_size = test_batch_size * 2

class TestDataGenerator(unittest.TestCase):
  def setUp(self):
    test_generator = train_dataset.batch(test_batch_size)
    self.pool_test_data = PoolData(test_pool_size, test_batch_size)

    iterator = iter(test_generator)
    img, first_kernel, second_kernel, sinc_kernel = next(iterator)
    gt_img, lq_img = feed_data(img, first_kernel, second_kernel, sinc_kernel, [feed_props_1, feed_props_2])

    self.img = img
    self.first_kernel = first_kernel
    self.second_kernel = second_kernel
    self.sinc_kernel = sinc_kernel

    self.gt_img = gt_img
    self.lq_img = lq_img

    self.first_pool_gt_shape = (4, 256, 256, 3)
    self.first_pool_lq_shape = (4, 64, 64, 3)

    self.second_pool_gt_shape = (8, 256, 256, 3)
    self.second_pool_lq_shape = (8, 64, 64, 3)

  def test_gt_image_stats(self):
    np_gt_img = self.gt_img.numpy()
    t_min = np_gt_img.min()
    t_max = np_gt_img.max()

    self.assertTrue(t_min >= 0, t_min)
    self.assertTrue(t_max <= 1, t_min)

  def test_lq_image_stats(self):
    np_lq_img = self.lq_img.numpy()
    t_min = np_lq_img.min()
    t_max = np_lq_img.max()
    
    self.assertTrue(t_min >= 0, t_min)
    self.assertTrue(t_max <= 1, t_min)

  def test_sinc_kernel_stats(self):
    np_kernel = self.sinc_kernel.numpy()
    t_min = np_kernel.min()
    t_max = np_kernel.max()
    
    self.assertTrue(t_min >= -1, t_min)
    self.assertTrue(t_max <= 1, t_min)

  def test_first_kernel_stats(self):
    np_kernel = self.first_kernel.numpy()
    t_min = np_kernel.min()
    t_max = np_kernel.max()
    
    self.assertTrue(t_min >= -1, t_min)
    self.assertTrue(t_max <= 1, t_min)

  def test_second_kernel_stats(self):
    np_kernel = self.second_kernel.numpy()
    t_min = np_kernel.min()
    t_max = np_kernel.max()
    
    self.assertTrue(t_min >= -1, t_min)
    self.assertTrue(t_max <= 1, t_min)

  def test_pad_image_shape(self):
    self.assertEqual(self.img.shape, (test_batch_size, pad_img_size, pad_img_size, 3), self.img.shape)

  def test_gt_image_shape(self):
    self.assertEqual(self.gt_img.shape, (test_batch_size, final_gt_img_size, final_gt_img_size, 3), self.gt_img.shape)

  def test_lq_image_shape(self):
    self.assertEqual(self.lq_img.shape, (test_batch_size, final_lq_img_size, final_lq_img_size, 3), self.lq_img.shape)

  def test_kernel_shape(self):
    self.assertEqual(self.sinc_kernel.shape, (test_batch_size, kernel_size, kernel_size), self.sinc_kernel.shape)
    self.assertEqual(self.first_kernel.shape, (test_batch_size, kernel_size, kernel_size), self.first_kernel.shape)
    self.assertEqual(self.second_kernel.shape, (test_batch_size, kernel_size, kernel_size), self.second_kernel.shape)

  def test_pool_data(self):
    gt_img, lq_img = self.pool_test_data.get_pool_data(self.gt_img, self.lq_img)

    self.assertEqual(self.pool_test_data.queue_gt.shape, self.first_pool_gt_shape)
    self.assertEqual(self.pool_test_data.queue_lr.shape, self.first_pool_lq_shape)

    gt_img, lq_img = self.pool_test_data.get_pool_data(gt_img, lq_img)

    self.assertEqual(self.pool_test_data.queue_gt.shape, self.second_pool_gt_shape)
    self.assertEqual(self.pool_test_data.queue_lr.shape, self.second_pool_lq_shape)

    gt_img, lq_img = self.pool_test_data.get_pool_data(gt_img, lq_img)

    self.assertEqual(self.pool_test_data.queue_gt.shape, self.second_pool_gt_shape)
    self.assertEqual(self.pool_test_data.queue_lr.shape, self.second_pool_lq_shape)

if __name__ == "__main__" and not using_notebook:
  unittest.main()

