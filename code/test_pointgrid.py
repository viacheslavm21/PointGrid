from io import StringIO
import sys
import unittest
import numpy as np
from network import *
import tensorflow as tf
import os
from scipy.io import savemat

class Test_Network(unittest.TestCase):

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
            
    def test_pc2voxel(self):
        pc = np.array([[113.772   , 108.51    ,  16.5534  ],
                       [113.775   ,  99.6067  ,   0.120465],
                       [113.772   ,  99.3912  ,   0.240929],
                       [113.772   , 108.51    ,  16.5534  ],
                       [113.775   ,  99.6067  ,   0.120465],
                       [113.772   ,  99.3912  ,   0.240929],
                       [113.772   , 108.51    ,  16.5534  ],
                       [113.775   ,  99.6067  ,   0.120465],
                       [113.772   ,  99.3912  ,   0.240929],
                       [113.772   , 108.51    ,  16.5534  ],
                       [113.775   ,  99.6067  ,   0.120465],
                       [113.772   ,  99.3912  ,   0.240929],
                       [113.772   , 108.51    ,  16.5534  ],
                       [113.775   ,  99.6067  ,   0.120465],
                       [113.772   ,  99.3912  ,   0.240929],
                       [113.772   , 108.51    ,  16.5534  ]])
        pc_label = np.zeros((16, NUM_SEG_PART))
        pc_label[:, 0] = 1
        data, label, index = pc2voxel(pc, pc_label)
        data_shape = np.array([N, N, N, 13])
        label_shape = np.array([N, N, N, K+1, NUM_SEG_PART])
        index_shape = np.array([N, N, N, K])

        self.assertTrue(np.allclose(data_shape, data.shape, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose(label_shape, label.shape, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose(index_shape, index.shape, rtol=1e-05, atol=1e-08))
        
        # invalid input
        with self.assertRaises(IndexError):
            pc2voxel(pc[0], pc_label[0])



class Test_Train_Test(unittest.TestCase):
    
    def test_1train_one_epoch_one_file(self):
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
    
    def test_2test_with_categoryfile(self):
        os.system("mv ../data/ShapeNet/test ../data/ShapeNet/test_old")
        os.system("mv ../category_file ../category_file_old")
        os.system("mkdir ../data/ShapeNet/test")
        with open("../category_file", "w") as outfile:
            outfile.write("\n".join(["first_category"]))
        outfile.close()
        randpoints = np.array([[113.772, 108.51, 16.5534], [113.775, 99.6067, 0.120465], [113.772, 99.3912, 0.240929],
                       [113.772, 108.51, 16.5534], [113.775, 99.6067, 0.120465], [113.772, 99.3912, 0.240929],
                       [113.772, 108.51, 16.5534], [113.775, 99.6067, 0.120465], [113.772, 99.3912, 0.240929],
                       [113.772, 108.51, 16.5534], [113.775, 99.6067, 0.120465], [113.772, 99.3912, 0.240929],
                       [113.772, 108.51, 16.5534], [113.775, 99.6067, 0.120465], [113.772, 99.3912, 0.240929],
                       [113.772, 108.51, 16.5534], ])
        labels = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0])
        category = 0
        new_filename = "../data/ShapeNet/test/test_data_0.mat"
        savemat(new_filename, {'points':randpoints,'labels':labels,'category':category})
        os.system("python -W ignore test.py")
        os.system("rm -r ../data/ShapeNet/test")
        os.system("rm ../category_file")
        os.system("mv ../data/ShapeNet/test_old ../data/ShapeNet/test")
        os.system("mv ../category_file_old ../category_file")
        
    def test_2test_without_categoryfile(self):
        os.system("mv ../data/ShapeNet/test ../data/ShapeNet/test_old")
        os.system("mv ../category_file ../category_file_old")
        os.system("mkdir ../data/ShapeNet/test")
        randpoints = np.array([[113.772, 108.51, 16.5534], [113.775, 99.6067, 0.120465], [113.772, 99.3912, 0.240929],
                       [113.772, 108.51, 16.5534], [113.775, 99.6067, 0.120465], [113.772, 99.3912, 0.240929],
                       [113.772, 108.51, 16.5534], [113.775, 99.6067, 0.120465], [113.772, 99.3912, 0.240929],
                       [113.772, 108.51, 16.5534], [113.775, 99.6067, 0.120465], [113.772, 99.3912, 0.240929],
                       [113.772, 108.51, 16.5534], [113.775, 99.6067, 0.120465], [113.772, 99.3912, 0.240929],
                       [113.772, 108.51, 16.5534], ])
        labels = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0])
        category = 0
        new_filename = "../data/ShapeNet/test/test_data_0.mat"
        savemat(new_filename, {'points':randpoints,'labels':labels,'category':category})
        os.system("python -W ignore test.py")
        os.system("rm -r ../data/ShapeNet/test")
        os.system("mv ../data/ShapeNet/test_old ../data/ShapeNet/test")
        os.system("mv ../category_file_old ../category_file")


if __name__ == '__main__':
    unittest.main()
