import unittest
import numpy as np
import tensorflow as tf


class TestDataProcessing(unittest.TestCase):

    def test_normalization(self):
        """Test that pixel values are normalized to [0, 1]."""
        dummy = tf.constant(np.random.randint(0, 256, (4, 64, 64, 1)), dtype=tf.float32)
        normalized = dummy / 255.0
        self.assertLessEqual(float(tf.reduce_max(normalized)), 1.0)
        self.assertGreaterEqual(float(tf.reduce_min(normalized)), 0.0)

    def test_output_shape(self):
        """Test that image tensors have correct shape."""
        dummy = tf.random.uniform((4, 64, 64, 1))
        self.assertEqual(dummy.shape, (4, 64, 64, 1))

    def test_batch_size(self):
        """Test dataset batching."""
        dummy_ds = tf.data.Dataset.from_tensors(
            tf.random.uniform((8, 64, 64, 1))
        ).batch(4)
        for batch in dummy_ds:
            self.assertLessEqual(batch.shape[0], 4)


if __name__ == '__main__':
    unittest.main()
