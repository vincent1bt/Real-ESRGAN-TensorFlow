import tensorflow as tf

def compute_l1_loss(y_true, y_fake):
  return tf.reduce_mean(tf.math.abs(y_true - y_fake))

def compute_perceptual_loss(y_list, y_trans_list):
  vgg_layers_weights = [0.1, 0.1, 1, 1, 1]
  loss = 0

  for feature_map_y, feature_map_y_trans, layer_weight in zip(y_list, y_trans_list, vgg_layers_weights):
    loss += tf.reduce_mean(tf.math.abs(feature_map_y - feature_map_y_trans)) * layer_weight
  
  return loss

def compute_gan_loss(real_logits, generated_logits, generator_loss_weight=1e-1, from_logits=True):
  real_labels = tf.ones_like(real_logits)
  generated_labels = tf.zeros_like(real_logits)

  generator_loss = tf.keras.losses.binary_crossentropy(
      real_labels, generated_logits, from_logits=from_logits
  )
  
  disc_real_loss = tf.keras.losses.binary_crossentropy(
      real_labels, real_logits, from_logits=from_logits
  )

  disc_fake_loss = tf.keras.losses.binary_crossentropy(
      generated_labels, generated_logits, from_logits=from_logits
  )

  discriminator_loss = disc_real_loss + disc_fake_loss

  return tf.reduce_mean(generator_loss) * generator_loss_weight, tf.reduce_mean(discriminator_loss)

