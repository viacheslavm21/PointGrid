# PointGrid: A Deep Network for 3D Shape Understanding

## Prerequisites:
### Requirements
1. Python 3.6.12 (with necessary common libraries such as numpy, scipy, etc.)
2. TensorFlow 1.13.2

A docker image with the necessary environment is available. To download it use:
<pre>
docker pull ogoldobina/point_grid:v1
</pre>
However, this image doesn't contain any code from this repo and is used only in our workflow.

### Input data format
You need to prepare your data in *.mat file with the following format:
- 'points': N x 3 array (x, y, z coordinates of the point cloud)
- 'labels': N x 1 array (1-based integer per-point labels)
- 'category': scalar (0-based integer model category)

## Quickstart
To clone this repo to your machine use:
<pre>
git clone https://github.com/viacheslavm21/PointGrid.git
</pre>

You may try to install prerequisites by following command: (from ./codes)
<pre>
python setup.py install
</pre>

However, **we highly recommend you to use docker container**. There are two options to do so:
1. You can download a docker image by running
<pre>
docker pull ogoldobina/point_grid:v2
</pre>
2. You can build an image yourself by running
<pre>
docker build https://github.com/viacheslavm21/PointGrid.git
</pre>
After that you can run the container
<pre>
docker run < image name >
</pre>

The repo provides four entry-points.
1. download.sh - downloads ModelNet40 dataset and unzips it.
<pre>
bash download.sh
</pre>
2. prepare.py - sampling points from triangle mesh, saving in .mat format.
<pre>
python prepare.py
</pre>
3. code/train.py - train the network.
<pre>
python code/train.py
</pre>
4. code/test.py - testing and evaluation. 
<pre>
python code/test.py
</pre>

Run the scripts sequentially to reproduce the results. 
Note: data preparation is a long process (a file in original dataset is triangular mesh, when the project needs pointclouds). Therefore, we prepared train and test data for you in the repo. You may either try download.sh + prepare.py. Or just start with train and test.

You can also run this workflow to perform all the steps sequentially:
<pre>
https://github.com/viacheslavm21/PointGrid/actions/workflows/main.yml
</pre>

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

Original repo: https://github.com/trucleduc/PointGrid
