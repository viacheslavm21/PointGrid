from io import StringIO
import sys
import unittest
from unittest.mock import patch
import numpy as np
from network import *
import tensorflow as tf
import os
from scipy.io import savemat

class TestNetwork(unittest.TestCase):

    def test_leak_relu(self):
        x = 1.0
        output = leak_relu(x)
        self.assertIsInstance(output, tf.Tensor)

    def test_integer_label_to_one_hot_label(self):

        integer_label = 1
        with self.assertRaises(AttributeError):
            integer_label_to_one_hot_label(integer_label)

        integer_label = np.array([1])
        one_hot = integer_label_to_one_hot_label(integer_label)
        expected_output = np.zeros((integer_label.shape[0], NUM_SEG_PART))
        expected_output[0, 1] = 1
        self.assertTrue(np.allclose(one_hot, expected_output, rtol=1e-05, atol=1e-08))

        integer_label = np.zeros((2,3))
        with self.assertRaises(AssertionError):
            integer_label_to_one_hot_label(integer_label)



class TestTrain(unittest.TestCase):
    
    def test_train_one_epoch_one_file(self):
        os.system("mv ../data/ShapeNet/train ../data/ShapeNet/train_old")
        os.system("mkdir ../data/ShapeNet/train")
        randpoints = np.array([[113.772, 108.51, 16.5534], [113.775, 99.6067, 0.120465], [113.772, 99.3912, 0.240929],
                       [113.772, 108.51, 16.5534], [113.775, 99.6067, 0.120465], [113.772, 99.3912, 0.240929],
                       [113.772, 108.51, 16.5534], [113.775, 99.6067, 0.120465], [113.772, 99.3912, 0.240929],
                       [113.772, 108.51, 16.5534], [113.775, 99.6067, 0.120465], [113.772, 99.3912, 0.240929],
                       [113.772, 108.51, 16.5534], [113.775, 99.6067, 0.120465], [113.772, 99.3912, 0.240929],
                       [113.772, 108.51, 16.5534], ])
        labels = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0])
        category = 0
        new_filename = "../data/ShapeNet/train/train_data_0.mat"
        savemat(new_filename, {'points':randpoints,'labels':labels,'category':category})
        os.system("python -W ignore train.py --epoch 1 --batch 1")
        os.system("rm -r ../data/ShapeNet/train")
        os.system("mv ../data/ShapeNet/train_old ../data/ShapeNet/train")
        
# class TestTest(unittest.TestCase):
    
#     def test_test(self):

#         os.system("python -W ignore test.py") 


if __name__ == '__main__':
    unittest.main()
