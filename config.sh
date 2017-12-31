#!/bin/bash

sudo apt-get update
sudo apt-get install tmux

conda create -n env_pytorch --clone="/opt/conda/envs/pytorch-py35"
source activate env_pytorch
conda install jupyter pymongo matplotlib -y
conda install -c anaconda mkl -y
conda install pytorch torchvision -c pytorch -y
