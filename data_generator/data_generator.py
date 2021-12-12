import tensorflow as tf
import random

from data_generator.data_utils import kernel_props_1, kernel_props_2, final_sinc_prob
from data_generator.data_utils import generate_sinc_kernel, generate_kernel
from data_generator.data_utils import random_add_gaussian_noise, random_add_poisson_noise
from data_generator.data_utils import filter2D, USMSharp

def augment_image(img, rotation=False):
  if tf.random.uniform([]) > 0.5:
    img = tf.image.flip_left_right(img)

  if rotation and tf.random.uniform([]) > 0.5:
    img = tf.image.flip_up_down(img)

  if rotation and tf.random.uniform([]) > 0.5:
    img = tf.image.rot90(img)

  return img

def load_image(image_path, crop_pad_size=400):
  img = tf.io.read_file(image_path)
  img = tf.image.decode_png(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  img = augment_image(img)

  size = tf.shape(img)
  height, width = size[0], size[1]

  if height < crop_pad_size or width < crop_pad_size:
    pad_h = tf.maximum(0, crop_pad_size - height)
    pad_w = tf.maximum(0, crop_pad_size - width)
              
        # height (top and bottom), width (left and right),   channels
    padding = [[0, pad_h], [0, pad_w], [0, 0]]
    img = tf.pad(img, padding, "REFLECT") 

  size = tf.shape(img)
  height, width = size[0], size[1]

  if height > crop_pad_size or width > crop_pad_size:
    if (height - crop_pad_size) <= 0:
      top = 0
    else:
      top = tf.random.uniform([], 0, height - crop_pad_size, dtype=tf.dtypes.int32)

    if (width - crop_pad_size) <= 0:
      left = 0
    else:
      left = tf.random.uniform([], 0, width - crop_pad_size, dtype=tf.dtypes.int32)
    
    img = tf.image.crop_to_bounding_box(img, top, left, crop_pad_size, crop_pad_size)

  return img

def load_test_img(image_path):
  img = tf.io.read_file(image_path)
  img = tf.image.decode_png(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  return img

def load_train_image(final_sinc_prob, kernel_props_1, kernel_props_2):
  def load_image_kernels(image_path):
    img = load_image(image_path)

    first_kernel = generate_kernel(kernel_props_1)
    second_kernel = generate_kernel(kernel_props_2)

    sinc_kernel = generate_sinc_kernel(final_sinc_prob)

    return img, first_kernel, second_kernel, sinc_kernel
    
  return load_image_kernels

load_function = load_train_image(final_sinc_prob, kernel_props_1, kernel_props_2)

feed_props_1 = {
  "resize_prob": [0.2, 0.7, 0.1],
  "resize_range": [0.15, 1.5],
  "gray_noise_prob": 0.4,
  "gaussian_noise_prob": 0.5,
  "noise_range": [1, 30],
  "poisson_scale_range": [0.05, 3],
}

feed_props_2 = {
  "resize_prob": [0.3, 0.4, 0.3],
  "resize_range": [0.3, 1.2],
  "gray_noise_prob": 0.4,
  "gaussian_noise_prob": 0.5,
  "noise_range": [1, 25],
  "poisson_scale_range": [0.05, 2.5],
}

def random_crop(imgs, out, final_size=256, scale=4):
  _, h_lq, w_lq, _ = out.shape

  lq_patch_size = final_size // scale # 64

  top = random.randint(0, h_lq - lq_patch_size)
  left = random.randint(0, w_lq - lq_patch_size)
  out = tf.image.crop_to_bounding_box(out, top, left, lq_patch_size, lq_patch_size)

  top_gt, left_gt = int(top * scale), int(left * scale)
  imgs = tf.image.crop_to_bounding_box(imgs, top_gt, left_gt, final_size, final_size)        

  return imgs, out

def degradation(imgs, kernels, opts_dict, blur_prob=1.0):
  if (blur_prob == 1.0) or ( tf.random.uniform([]) < blur_prob):
    imgs = filter2D(imgs, kernels)
  
  updown_type = random.choices(['up', 'down', 'keep'], opts_dict['resize_prob'])[0]

  if updown_type == 'up':
    scale = tf.random.uniform([], 1, opts_dict['resize_range'][1])
  elif updown_type == 'down':
    scale = tf.random.uniform([], opts_dict['resize_range'][0], 1)
  else:
    scale = 1

  mode = random.choice(['area', 'bilinear', 'bicubic'])

  if scale != 1:
    size = int(scale * imgs.shape[1])
    imgs = tf.image.resize(imgs, [size, size], method=mode)

  gray_noise_prob = opts_dict['gray_noise_prob']

  if tf.random.uniform([]) < opts_dict['gaussian_noise_prob']:
    imgs = random_add_gaussian_noise(imgs, sigma_range=opts_dict['noise_range'], gray_prob=gray_noise_prob)
  else:
    imgs = random_add_poisson_noise(imgs, scale_range=opts_dict['poisson_scale_range'], gray_prob=gray_noise_prob)
  
  return imgs

usm_sharpener = USMSharp()

def feed_data(imgs, first_kernels, second_kernels, sinc_kernels, feed_props, final_size=256, jpg_quality=(30, 95)):
  usm_imgs = usm_sharpener.sharp(imgs)
  batch, height, width, _ = usm_imgs.shape
  scale = 4

  out = degradation(usm_imgs, first_kernels, feed_props[0], blur_prob=1.0)

  out = tf.clip_by_value(out, 0, 1)
  out = [tf.image.random_jpeg_quality(out[i], jpg_quality[0], jpg_quality[1]) for i in range(0, batch)]

  out = tf.convert_to_tensor(out)

  out = degradation(out, second_kernels, feed_props[1], blur_prob=0.8)

  if tf.random.uniform([]) < 0.5:
    # resize back + the final sinc filter
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    size = height // scale
    out = tf.image.resize(out, [size, size], method=mode)
    out = filter2D(out, sinc_kernels)
    
    # JPEG compression
    out = tf.clip_by_value(out, 0, 1)
    out = [tf.image.random_jpeg_quality(out[i], jpg_quality[0], jpg_quality[1]) for i in range(0, batch)]
    out = tf.convert_to_tensor(out)

  else:
    # JPEG compression
    out = tf.clip_by_value(out, 0, 1)
    out = [tf.image.random_jpeg_quality(out[i], jpg_quality[0], jpg_quality[1]) for i in range(0, batch)]
    out = tf.convert_to_tensor(out)
    # resize back + the final sinc filter
    
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    size = height // scale
    out = tf.image.resize(out, [size, size], method=mode)
    out = filter2D(out, sinc_kernels)


  out = tf.clip_by_value(tf.math.round(out * 255.0), 0, 255) / 255.

  imgs, out = random_crop(imgs, out, final_size=final_size, scale=scale)
  
  return imgs, out

class PoolData():
  def __init__(self, pool_size, batch_size):
    self.pool_size = pool_size
    self.idx = list(range(self.pool_size))
    self.queue_gt = None
    self.queue_lr = None
    self.batch_size = batch_size

    if not pool_size % batch_size == 0:
      raise TypeError(
        f"pool_size ({pool_size}) % batch_size ({batch_size}) should be 0"
      )
  
  def get_pool_data(self, new_imgs, new_out):
    if self.queue_gt == None:
      self.queue_gt = new_imgs
      self.queue_lr = new_out

      return new_imgs, new_out

    elif self.queue_gt.shape[0] == self.pool_size:
      self.idx = tf.random.shuffle(self.idx)

      # shuffle
      self.queue_gt = tf.gather(self.queue_gt, self.idx)
      self.queue_lr = tf.gather(self.queue_lr, self.idx)

      o_new_imgs = self.queue_gt[0:self.batch_size]
      o_new_out = self.queue_lr[0:self.batch_size]

      self.queue_gt = tf.concat([new_imgs, self.queue_gt[self.batch_size:]], axis=0)
      self.queue_lr = tf.concat([new_out, self.queue_lr[self.batch_size:]], axis=0)
      
      assert self.queue_gt.shape[0] == self.pool_size

      return o_new_imgs, o_new_out

    else:
      self.queue_gt = tf.concat([self.queue_gt, new_imgs], axis=0)
      self.queue_lr = tf.concat([self.queue_lr, new_out], axis=0)

      return new_imgs, new_out

