#!/usr/bin/env bash

# Install the required libraries
pip install numpy scipy h5py
pip install scikit-learn Pillow imutils
pip install beautifulsoup4
pip install tensorflow-gpu
pip install keras
pip install opencv-contrib-python

# Install Retinanet
cd ~
git clone https://github.com/fizyr/keras-retinanet
cd keras-retinanet
git checkout 42068ef9e406602d92a1afe2ee7d470f7e9860df
python setup.py install
