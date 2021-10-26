# PointGrid: A Deep Network for 3D Shape Understanding

## Prerequisites:
1. Python (with necessary common libraries such as numpy, scipy, etc.)
2. TensorFlow 1.13.2
3. You need to prepare your data in *.mat file with the following format:
	- 'points': N x 3 array (x, y, z coordinates of the point cloud)
	- 'labels': N x 1 array (1-based integer per-point labels)
	- 'category': scalar (0-based integer model category)

## Quickstart
For cloning this repo to your machine use:
<pre>
git clone https://github.com/viacheslavm21/PointGrid.git
</pre>

You may try to install prerequisites by following command: (from /root/codes)
<pre>
python setup.py install
</pre>

However, we highly recommend you to build docker image from provided Dockerfile:
<pre>
docker build -f Dockerfile .
docker run < image name >
</pre>

Docker image will contain cloned repository.

The repo provides four entry-points.
1. /root/download.sh - downloads ModelNet40 dataset and unzips it.
<pre>
bash /root/download.sh
</pre>
2. /root/prepare.py - sampling points from triangle mesh, saving in .mat format.
<pre>
python /root/prepare.py
</pre>
3. /root/code/train.py - train the network.
<pre>
python /root/train.py
</pre>
4. /root/code/test.py - testing and evaluation. 
<pre>
python /root/test.py
</pre>

Run the scripts sequentially to reproduce the results.

## Development 

In network.py change NUM_CATEGORY from 40 to n, where n is the number of categories in your dataset (if you want to try another dataset)

## Run tests

In test_pointgrid.py there are unittests for modules from network.py and for train and test from code directory. You can run all tests with command:
<pre>
cd code
python -W ignore -m unittest test_pointgrid.py -v
</pre>
In this command warnings are disabled as in the project the old packets are used.

## Original article 

If you find this code useful, please cite our work at <br />
<pre>
@article{PointGrid,
	author = {Truc Le and Ye Duan},
	titile = {{PointGrid: A Deep Network for 3D Shape Understanding}},
	journal = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = {June},
	year = {2018},
}
</pre>
