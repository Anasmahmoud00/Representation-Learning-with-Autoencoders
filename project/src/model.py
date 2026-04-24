import tensorflow as tf
from tensorflow.keras import layers, models


def build_ae(latent_dim=32):
    """Build Autoencoder (AE) model."""
    # Encoder
    encoder_inputs = layers.Input(shape=(64, 64, 1))
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(encoder_inputs)
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, name="latent_space")(x)
    encoder = models.Model(encoder_inputs, latent, name="Encoder")

    # Decoder
    decoder_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(16 * 16 * 64, activation='relu')(decoder_inputs)
    x = layers.Reshape((16, 16, 64))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    outputs = layers.Conv2DTranspose(1, 3, padding='same', activation='sigmoid')(x)
    decoder = models.Model(decoder_inputs, outputs, name="Decoder")

    # Full AE
    ae = models.Model(encoder_inputs, decoder(encoder(encoder_inputs)), name="AE")
    return ae, encoder, decoder


def build_vae(latent_dim=32):
    """Build Variational Autoencoder (VAE) model."""
    # Encoder
    inputs = layers.Input(shape=(64, 64, 1))
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])
    encoder = models.Model(inputs, [z_mean, z_log_var, z], name="VAE_Encoder")

    # Decoder
    decoder_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(16 * 16 * 64, activation='relu')(decoder_inputs)
    x = layers.Reshape((16, 16, 64))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    outputs = layers.Conv2DTranspose(1, 3, padding='same', activation='sigmoid')(x)
    decoder = models.Model(decoder_inputs, outputs, name="VAE_Decoder")

    return encoder, decoder
