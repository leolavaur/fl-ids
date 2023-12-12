"""Eiffel utilities."""

from tensorflow import keras
import tensorflow as tf


def set_seed(seed: int) -> None:
    """Set the random seed.

    The seed is set for NumPy, Python's random module, and TensorFlow.

    Parameters
    ----------
    seed : int
        The seed to use for random number generation.
    """
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
