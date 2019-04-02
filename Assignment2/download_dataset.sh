#!/bin/bash
# script for downloading the dataset

cd data
wget http://miniplaces.csail.mit.edu/data/data.tar.gz
wget https://raw.githubusercontent.com/CSAILVision/miniplaces/master/data/train.txt
wget https://raw.githubusercontent.com/CSAILVision/miniplaces/master/data/val.txt

tar -xzf data.tar.gz
rm ./data.tar.gz
cd ..