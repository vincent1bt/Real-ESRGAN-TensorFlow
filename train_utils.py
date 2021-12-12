import tensorflow as tf

from networks.loss_functions import compute_l1_loss, compute_gan_loss, compute_perceptual_loss

def no_gan_inner_step(gt_images,
                      lq_images,
                      main_model,
                      optimizer):
  with tf.GradientTape() as tape:
    generated_images = main_model(lq_images)
    loss = compute_l1_loss(gt_images, generated_images)

  gradients = tape.gradient(loss, main_model.trainable_variables)

  optimizer.apply_gradients(zip(gradients, main_model.trainable_variables))

  return loss

def gan_inner_step(gt_images, usm_gt_images, lq_images, generator, discriminator, vgg_model, gen_opt, disc_opt):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(lq_images)
    l1_loss = compute_l1_loss(usm_gt_images, generated_images)

    true_features = vgg_model(usm_gt_images)
    fake_features = vgg_model(tf.clip_by_value(generated_images, 0.0, 1.0))

    perceptual_loss = compute_perceptual_loss(true_features, fake_features)

    true_logits = discriminator(gt_images)
    fake_logits = discriminator(generated_images)

    gen_loss, disc_loss = compute_gan_loss(true_logits, fake_logits)

    total_generator_loss = gen_loss + perceptual_loss + l1_loss

  generator_gradients = gen_tape.gradient(total_generator_loss, generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  gen_opt.apply_gradients(zip(generator_gradients, generator.trainable_variables))
  disc_opt.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

  return total_generator_loss, gen_loss, perceptual_loss, l1_loss, disc_loss

class AverageModelWeights():
  def __init__(self, model, weights, decay=0.999):
    self.ema_model = model
    self.decay = decay

    self.ema_model.set_weights(weights) 
  
  def compute_ema_weights(self, normal_model):
    normal_weights = normal_model.get_weights()
    weights = self.ema_model.get_weights()

    weights_list = [weight * self.decay + normal_weight * (1 - self.decay) for normal_weight, weight in zip(normal_weights, weights)]

    self.ema_model.set_weights(weights_list)

# def lr_decay(optimizer, learning_rate):
#   current_lr = optimizer.lr
#   new_lr = current_lr - (learning_rate / 8000)
#   optimizer.lr = new_lr

#   return optimizer

