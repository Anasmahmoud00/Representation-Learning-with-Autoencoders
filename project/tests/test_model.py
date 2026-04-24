import unittest
import tensorflow as tf
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.model import build_ae, build_vae


class TestAEModel(unittest.TestCase):

    def setUp(self):
        self.ae, self.encoder, self.decoder = build_ae(latent_dim=32)
        self.dummy = tf.random.uniform((2, 64, 64, 1))

    def test_ae_output_shape(self):
        """AE output should match input shape."""
        out = self.ae(self.dummy)
        self.assertEqual(out.shape, (2, 64, 64, 1))

    def test_encoder_latent_shape(self):
        """Encoder should produce correct latent dimension."""
        latent = self.encoder(self.dummy)
        self.assertEqual(latent.shape, (2, 32))

    def test_decoder_output_shape(self):
        """Decoder should reconstruct original image shape."""
        latent = tf.random.uniform((2, 32))
        out = self.decoder(latent)
        self.assertEqual(out.shape, (2, 64, 64, 1))


class TestVAEModel(unittest.TestCase):

    def setUp(self):
        self.enc, self.dec = build_vae(latent_dim=32)
        self.dummy = tf.random.uniform((2, 64, 64, 1))

    def test_vae_encoder_output_count(self):
        """VAE encoder should return 3 tensors: z_mean, z_log_var, z."""
        outputs = self.enc(self.dummy)
        self.assertEqual(len(outputs), 3)

    def test_vae_encoder_z_mean_shape(self):
        z_mean, z_log_var, z = self.enc(self.dummy)
        self.assertEqual(z_mean.shape, (2, 32))

    def test_vae_decoder_output_shape(self):
        z_sample = tf.random.normal((2, 32))
        out = self.dec(z_sample)
        self.assertEqual(out.shape, (2, 64, 64, 1))


if __name__ == '__main__':
    unittest.main()
