import tensorflow as tf
import argparse
import builtins
import os
from glob import glob

from data_generator.data_generator import load_test_img

parser = argparse.ArgumentParser()

parser.add_argument('--use_ema_model', default=False, action='store_true', help='Use ema model')
parser.add_argument('--batch_size', type=int, default=4, help='Number batches')

using_notebook = getattr(builtins, "__IPYTHON__", False)

opts = parser.parse_args([]) if using_notebook else parser.parse_args()

batch_size = opts.batch_size

if opts.use_ema_model:
    model = tf.keras.models.load_model('models/ema_gan_model')
else:
    model = tf.keras.models.load_model('models/gan_model')

data_path = os.path.abspath("./data/test_shots/*.png")
test_images_paths = sorted(glob(data_path))

test_dataset = tf.data.Dataset.from_tensor_slices((test_images_paths))
test_dataset = test_dataset.shuffle(len(test_images_paths))
test_dataset = test_dataset.map(load_test_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

test_generator = test_dataset.batch(batch_size)

for img in test_generator:
    output = model(img)

    for index, o_img in enumerate(output):
        tf.keras.utils.save_img(f"data/final_images/image{index}.png", o_img)
