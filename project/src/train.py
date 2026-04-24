import numpy as np
import tensorflow as tf


@tf.function
def train_step_vae(images, encoder, decoder, optimizer):
    """Single training step for VAE with KL + reconstruction loss."""
    with tf.GradientTape() as tape:
        z_mean, z_log_var, z = encoder(images)
        reconstruction = decoder(z)

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(images - reconstruction), axis=[1, 2, 3])
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
            )
        )
        total_loss = reconstruction_loss + kl_loss

    grads = tape.gradient(
        total_loss,
        encoder.trainable_variables + decoder.trainable_variables
    )
    optimizer.apply_gradients(zip(
        grads,
        encoder.trainable_variables + decoder.trainable_variables
    ))
    return total_loss, reconstruction_loss, kl_loss


def train_vae(vae_encoder, vae_decoder, train_ds, epochs=5):
    """Full VAE training loop."""
    optimizer = tf.keras.optimizers.Adam()
    for epoch in range(epochs):
        total_loss_avg = []
        for batch, _ in train_ds:
            loss, rec_loss, kl = train_step_vae(batch, vae_encoder, vae_decoder, optimizer)
            total_loss_avg.append(loss)
        print(f"Epoch {epoch+1}: Loss = {np.mean(total_loss_avg):.4f}")
    return vae_encoder, vae_decoder
