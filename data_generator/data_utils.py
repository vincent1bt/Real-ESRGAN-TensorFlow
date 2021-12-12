import tensorflow as tf
import math
import numpy as np
import random
import cv2

from tensorflow.keras import layers


def mesh_grid(kernel_size):
  ax = tf.range(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
  xx1, yy1 = tf.meshgrid(ax, ax)

  xx = tf.reshape(xx1, (kernel_size * kernel_size, 1))
  yy = tf.reshape(yy1, (kernel_size * kernel_size, 1))

  xy = tf.stack([xx, yy], axis=1)

  xy = tf.reshape(xy, (kernel_size, kernel_size, 2))

  return xy, xx1, yy1

def sigma_matrix2(sig_x, sig_y, theta):
  d_matrix = [[sig_x**2, 0], [0, sig_y**2]]
  u_matrix = [[tf.cos(theta), -tf.sin(theta)], [tf.sin(theta), tf.cos(theta)]]

  return tf.matmul(u_matrix, tf.matmul(d_matrix, tf.transpose(u_matrix)))

def pdf2(sigma_matrix, grid):
  inverse_sigma = tf.linalg.inv(sigma_matrix)
  x = tf.reduce_sum((tf.matmul(grid, inverse_sigma) * grid), 2) * -0.5 
  kernel = tf.exp(x)

  return kernel

def filter2D(imgs, kernels):
  b, h, w, c = imgs.shape
  k = kernels.shape[-1]
  pad_size = k // 2
  padding = [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]]
  imgs = tf.pad(imgs, padding, "REFLECT")

  _, ph, pw, _ = imgs.shape

  imgs = tf.transpose(imgs, [1, 2, 3, 0]) # H x W x C x B
  imgs = tf.reshape(imgs, (1, ph, pw, c * b)) # 1 x H x W x B*C

  kernels = tf.transpose(kernels, [1, 2, 0]) # k, k, b
  kernels = tf.reshape(kernels, [k, k, 1, b]) # k, k, 1, b
  kernels = tf.repeat(kernels, repeats=[c], axis=-1) # k, k, 1, b * c
  
  # kernel_height, kernel_width, input_filters, output_filters
  conv = layers.Conv2D(b*c, k, weights=[kernels], use_bias=False, groups=b*c)
  conv.trainable = False

  imgs = conv(imgs)

  imgs = tf.reshape(imgs, (h, w, c, b)) # H x W x C x B
  imgs = tf.transpose(imgs, [3, 0, 1, 2]) # B x H x W x C

  return imgs

def gaussian_filter2d(imgs, filter):
  b, h, w, c = imgs.shape
  k = filter.shape[-1]
  pad_size = k // 2
  padding = [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]]
  imgs = tf.pad(imgs, padding, "REFLECT")

  _, ph, pw, _ = imgs.shape

  imgs = tf.transpose(imgs, [0, 3, 1, 2]) # B x C x H x W 
  imgs = tf.reshape(imgs, (b*c, ph, pw, 1)) # B*C x H x W x 1 

  filter = tf.expand_dims(filter, axis=-1)
  filter = tf.expand_dims(filter, axis=-1) # k, k, 1, 1

  # kernel_height, kernel_width, input_filters, output_filters
  conv = layers.Conv2D(1, k, weights=[filter], use_bias=False)
  conv.trainable = False

  imgs = conv(imgs)

  imgs = tf.reshape(imgs, (b, c, h, w)) # B x C x H x W 
  imgs = tf.transpose(imgs, [0, 2, 3, 1]) # B x H x W x C

  return imgs

class USMSharp():
  def __init__(self, filter_size=51):
    self.filter_size = 51
    self.sigma = 0.3 * ((filter_size - 1) * 0.5 - 1) + 0.8

    kernel = cv2.getGaussianKernel(filter_size, 0)
    self.kernel = tf.matmul(kernel, tf.transpose(kernel))
    self.threshold = 10
    self.weight = 0.5
  
  def sharp(self, imgs):
    blur = gaussian_filter2d(imgs, self.kernel)

    residual = imgs - blur

    mask = tf.math.abs(residual) * 255 > self.threshold
    mask = tf.cast(mask, dtype=tf.float32)

    soft_mask = gaussian_filter2d(mask, self.kernel)

    sharp = imgs + (self.weight * residual)

    sharp = tf.clip_by_value(sharp, 0, 1)

    return soft_mask * sharp + (1 - soft_mask) * imgs



def lowpass_kernel(x, y, cutoff, kernel_size):
  return cutoff * tf.math.special.bessel_j1(cutoff * tf.sqrt(
          (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)) / (2 * np.pi * tf.sqrt(
              (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2))

def inner_kernel_map(x1, y1, z, cutoff, kernel_size):
  def inner_func(arr):
    if arr[0] == x1 and arr[1] == y1:
      return z 
    else:
      return lowpass_kernel(arr[0], arr[1], cutoff, kernel_size)
  return inner_func

def circular_lowpass_kernel(cutoff, kernel_size, pad_to=0):
  assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'

  x = [[i] * kernel_size for i in range(kernel_size)]
  x = tf.cast(x, dtype=tf.float32)
  y = tf.transpose(x)

  xy = tf.stack([x, y], axis=2)
  xy = tf.reshape(xy, (-1, 2))

  y1 = (kernel_size - 1) // 2
  x1 = (kernel_size - 1) // 2

  z = cutoff ** 2 / (4 * np.pi)

  inner_func_ = inner_kernel_map(x1, y1, z, cutoff, kernel_size)

  # kernel = [[z if i == x1 and j == y1 else lowpass_kernel(i, j, cutoff, kernel_size) for i, j in zip(ia, ja)] for ia, ja in zip(x, y)]
  # kernel = [z if i == x1 and j == y1 else lowpass_kernel(i, j, cutoff, kernel_size) for i, j in xy]
  kernel = tf.map_fn(inner_func_, xy, fn_output_signature=tf.float32)
  kernel = tf.reshape(kernel, (kernel_size, kernel_size))
  
  kernel = kernel / tf.reduce_sum(kernel)
  
  if pad_to > kernel_size:
    pad_size = (pad_to - kernel_size) // 2
    padding = [[pad_size, pad_size], [pad_size, pad_size]]
    kernel = tf.pad(kernel, padding) 
    
  return kernel

def bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, isotropic=True):
  grid, _, _ = mesh_grid(kernel_size)

  if isotropic:
    sigma_matrix = [[sig_x**2, 0], [0, sig_x**2]]
    sigma_matrix = tf.convert_to_tensor(sigma_matrix, dtype=tf.float32)
  else:
    sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
  
  kernel = pdf2(sigma_matrix, grid)
  kernel = kernel / tf.reduce_sum(kernel)

  return kernel

def random_bivariate_Gaussian(kernel_size, kernel_props, isotropic=True):
  rotation_range = (-math.pi, math.pi)

  sigma_x_range = kernel_props["sigma_x_range"]
  sigma_y_range = kernel_props["sigma_y_range"]

  assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
  assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'

  sigma_x = tf.random.uniform([], sigma_x_range[0], sigma_x_range[1])

  if isotropic is False:
    assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
    assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
    
    sigma_y = tf.random.uniform([], sigma_y_range[0], sigma_y_range[1])
    rotation = tf.random.uniform([], rotation_range[0], rotation_range[1])
  else:
    sigma_y = sigma_x
    rotation = 0

  kernel = bivariate_Gaussian(kernel_size, sigma_x, sigma_y, rotation, isotropic=isotropic)

  return kernel / tf.reduce_sum(kernel)

def bivariate_generalized_Gaussian(kernel_size, sig_x, sig_y, theta, beta, isotropic=True):
  grid, _, _ = mesh_grid(kernel_size)

  if isotropic:
    sigma_matrix = [[sig_x**2, 0], [0, sig_x**2]]
    sigma_matrix = tf.convert_to_tensor(sigma_matrix, dtype=tf.float32)
  else:
    sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)

  inverse_sigma = tf.linalg.inv(sigma_matrix)

  x = tf.reduce_sum((tf.matmul(grid, inverse_sigma) * grid), 2)
  x = tf.pow(x, beta) * -0.5 
  kernel = tf.exp(x)

  return kernel / tf.reduce_sum(kernel)

def random_bivariate_generalized_Gaussian(kernel_size, kernel_props, isotropic=True):
  rotation_range = (-math.pi, math.pi)

  sigma_x_range = kernel_props["sigma_x_range"]
  sigma_y_range = kernel_props["sigma_y_range"]
  beta_range = kernel_props["betag_range"]

  assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
  assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'

  sigma_x = tf.random.uniform([], sigma_x_range[0], sigma_x_range[1])

  if isotropic is False:
    assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
    assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'

    sigma_y = tf.random.uniform([], sigma_y_range[0], sigma_y_range[1])
    rotation = tf.random.uniform([], rotation_range[0], rotation_range[1])
  else:
    sigma_y = sigma_x
    rotation = 0

  # assume beta_range[0] < 1 < beta_range[1]
  if tf.random.uniform([]) < 0.5:
    beta = tf.random.uniform([], beta_range[0], 1)
  else:
    beta = tf.random.uniform([], 1, beta_range[1])

  kernel = bivariate_generalized_Gaussian(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic)

  return kernel / tf.reduce_sum(kernel)

def bivariate_plateau(kernel_size, sig_x, sig_y, theta, beta, isotropic=True):
  grid, _, _ = mesh_grid(kernel_size)

  if isotropic:
    sigma_matrix = [[sig_x**2, 0], [0, sig_x**2]]
    sigma_matrix = tf.convert_to_tensor(sigma_matrix, dtype=tf.float32)
  else:
    sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)

  inverse_sigma = tf.linalg.inv(sigma_matrix)

  x = tf.reduce_sum((tf.matmul(grid, inverse_sigma) * grid), 2)
  x = tf.pow(x, beta) + 1
  kernel = tf.math.reciprocal(x)
  
  return kernel / tf.reduce_sum(kernel)

def random_bivariate_plateau(kernel_size, kernel_props, isotropic=True):
  rotation_range = (-math.pi, math.pi)

  sigma_x_range = kernel_props["sigma_x_range"]
  sigma_y_range = kernel_props["sigma_y_range"]
  beta_range = kernel_props["betap_range"]

  assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
  assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'

  sigma_x = tf.random.uniform([], sigma_x_range[0], sigma_x_range[1])

  if isotropic is False:
    assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
    assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'

    sigma_y = tf.random.uniform([], sigma_y_range[0], sigma_y_range[1])
    rotation = tf.random.uniform([], rotation_range[0], rotation_range[1])
  else:
      sigma_y = sigma_x
      rotation = 0

  if tf.random.uniform([]) < 0.5:
    beta = tf.random.uniform([], beta_range[0], 1)
  else:
    beta = tf.random.uniform([], 1, beta_range[1])

  kernel = bivariate_plateau(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic)

  return kernel / tf.reduce_sum(kernel)

def random_mixed_kernels(kernel_size, kernel_props):
  kernel_type = random.choices(kernel_props["kernel_list"], kernel_props["kernel_prob"])[0]

  if kernel_type == 'iso':
    kernel = random_bivariate_Gaussian(kernel_size, kernel_props, isotropic=True)
  elif kernel_type == 'aniso':
    kernel = random_bivariate_Gaussian(kernel_size, kernel_props, isotropic=False)
  elif kernel_type == 'generalized_iso':
    kernel = random_bivariate_generalized_Gaussian(kernel_size, kernel_props, isotropic=True)
  elif kernel_type == 'generalized_aniso':
    kernel = random_bivariate_generalized_Gaussian(kernel_size, kernel_props, isotropic=False)
  elif kernel_type == 'plateau_iso':
    kernel = random_bivariate_plateau(kernel_size, kernel_props, isotropic=True)
  elif kernel_type == 'plateau_aniso':
    kernel = random_bivariate_plateau(kernel_size, kernel_props, isotropic=False)
  
  return kernel





def generate_poisson_noise(img, scale=1.0, gray_noise=0):
  b, h, w, c = img.shape

  gray_noise = tf.reshape(gray_noise, (b, 1, 1, 1))
  cal_gray_noise = tf.reduce_sum(gray_noise) > 0

  base_2 = tf.math.log([2.])

  if cal_gray_noise:
    img_gray = tf.image.rgb_to_grayscale(img)
    img_gray = tf.clip_by_value(tf.math.round(img_gray * 255.0), 0, 255) / 255.

    vals_list = [len(tf.unique(tf.reshape(img_gray[i, :, :, :], h * w))[0]) for i in range(b)]

    vals_list = [2 ** tf.math.ceil(tf.math.log(tf.cast(vals, tf.float32)) / base_2) for vals in vals_list]

    vals = tf.reshape(vals_list, (b, 1, 1, 1))

    out = tf.random.poisson([], img_gray * vals) / vals
    noise_gray = out - img_gray
    noise_gray = tf.repeat(noise_gray, repeats=[3], axis=-1)

  # always calculate color noise
  # round and clip image for counting vals correctly
  img = tf.clip_by_value(tf.math.round(img * 255.0), 0, 255) / 255.

  # use for-loop to get the unique values for each sample
  vals_list = [len(tf.unique(tf.reshape(img[i, :, :, :], h * w * c))[0]) for i in range(b)]
  vals_list = [2 ** tf.math.ceil(tf.math.log(tf.cast(vals, tf.float32)) / base_2) for vals in vals_list]

  vals = tf.reshape(vals_list, (b, 1, 1, 1))

  out = tf.random.poisson([], img * vals) / vals
  noise = out - img

  if cal_gray_noise:
    noise = noise * (1 - gray_noise) + noise_gray * gray_noise

  scale = tf.reshape(scale, (b, 1, 1, 1))

  return noise * scale

def random_generate_poisson_noise(img, scale_range=(0, 1.0), gray_prob=0):
  scale = tf.random.uniform([img.shape[0]], dtype=img.dtype) * (scale_range[1] - scale_range[0]) + scale_range[0]

  gray_noise = tf.random.uniform([img.shape[0]], dtype=img.dtype)
  gray_noise = tf.cast((gray_noise < gray_prob), dtype=tf.float32)

  return generate_poisson_noise(img, scale, gray_noise)

def random_add_poisson_noise(img, scale_range=(0, 1.0), gray_prob=0):
  noise = random_generate_poisson_noise(img, scale_range, gray_prob)
  out = img + noise
  out = tf.clip_by_value(out, 0, 1)

  return out



def generate_gaussian_noise(img, sigma=10, gray_noise=0):
  b, h, w, c = img.shape

  sigma = tf.reshape(sigma, (b, 1, 1, 1))
  gray_noise = tf.reshape(gray_noise, (b, 1, 1, 1))
  cal_gray_noise = tf.reduce_sum(gray_noise) > 0

  if cal_gray_noise:
    noise_gray = tf.random.normal([1, h, w, 1], dtype=img.dtype) * sigma / 255.

  # always calculate color noise
  noise = tf.random.normal([b, h, w, c], dtype=img.dtype) * sigma / 255.

  if cal_gray_noise:
    noise = noise * (1 - gray_noise) + noise_gray * gray_noise

  return noise

def random_generate_gaussian_noise(img, sigma_range=(0, 10), gray_prob=0):
  sigma = tf.random.uniform([img.shape[0]], dtype=img.dtype) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
  
  # gray_noise = torch.rand(img.size(0), dtype=img.dtype, device=img.device)
  # gray_noise = (gray_noise < gray_prob).float()

  gray_noise = tf.random.uniform([img.shape[0]], dtype=img.dtype)
  gray_noise = tf.cast((gray_noise < gray_prob), dtype=tf.float32)

  return generate_gaussian_noise(img, sigma, gray_noise)

def random_add_gaussian_noise(img, sigma_range=(0, 1.0), gray_prob=0):
  noise = random_generate_gaussian_noise(img, sigma_range, gray_prob)
  
  out = img + noise
  out = tf.clip_by_value(out, 0, 1)

  return out



def generate_kernel(kernel_props):
  kernel_range = [2 * v + 1 for v in range(3, 11)] # from 7 to 21
  kernel_size = random.choice(kernel_range)

  sinc_prob = kernel_props["sinc_prob"]

  if tf.random.uniform([]) < sinc_prob:
    if kernel_size < 13:
      omega_c = tf.random.uniform([], np.pi / 3, np.pi)
    else:
      omega_c = tf.random.uniform([], np.pi / 5, np.pi)
      
    kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
  
  else:
    kernel = random_mixed_kernels(kernel_size, kernel_props)

  if kernel_size < 21:
    pad_to = 21
    pad_size = (pad_to - kernel_size) // 2
    padding = [[pad_size, pad_size], [pad_size, pad_size]]
    kernel = tf.pad(kernel, padding)

  return kernel

def generate_sinc_kernel(sinc_prob):
  kernel_range = [2 * v + 1 for v in range(3, 11)] # from 7 to 21

  if tf.random.uniform([]) < sinc_prob:
    kernel_size = random.choice(kernel_range)
    omega_c = tf.random.uniform([], np.pi / 3, np.pi, dtype=tf.float32)

    sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
  else:
    pulse = tf.constant([[1], [0]], dtype=tf.float32)
    pad_size = 10
    padding = [[pad_size, pad_size - 1], [pad_size, pad_size]]
    pulse = tf.pad(pulse, padding) 
    sinc_kernel = pulse

  return sinc_kernel

final_sinc_prob = 0.8

kernel_props_1 = {
    "kernel_list": ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    "kernel_prob": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    "sigma_x_range": [0.2, 3],
    "sigma_y_range": [0.2, 3],
    "betag_range": [0.5, 4],
    "betap_range": [1, 2],
    "sinc_prob": 0.1
}

kernel_props_2 = {
    "kernel_list": ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    "kernel_prob": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    "sigma_x_range": [0.2, 1.5],
    "sigma_y_range": [0.2, 1.5],
    "betag_range": [0.5, 4],
    "betap_range": [1, 2],
    "sinc_prob": 0.1
}

