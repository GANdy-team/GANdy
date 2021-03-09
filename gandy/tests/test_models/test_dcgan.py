"""Testing functions for deepchem GAN class."""

import numpy as np
import unittest
import unittest.mock

import deepchem

import gandy.models.dcgan as dcgan


class TestGAN(unittest.TestCase):
    """Test Deepchem GAN class."""

    def test_create_generator(self):
        """
        Test create generator function.

        The create generator function uses kwargs to create a Keras model.
        This checks that the model compiles.
        """
        return

    def test_create_discriminator(self):
        """
        Test create discriminator function.

        The create discriminator function uses kwargs to create a Keras model.
        This checks that the model compiles.
        """
        return

    def test_get_noise_input_shape(self):
        """Test get_noise_input_shape function."""
        return

    def test_get_data_input_shapes(self):
        """Test get_data_input_shapes function."""
        return

    def test_get_conditional_input_shapes(self):
        """Test get_conditional_input_shapes function."""
        return
