import numpy as np
import cv2
import torch


import unet
import nettrain as nt
import nettrainutils as ntu
import iterlogging as il


with open('Oxford pets/Segmentation/annotations/trainval.txt') as fin:
    train_names = [line.split()[0] for s in fin.readlines()]

with open('Oxford pets/Segmentation/annotations/test.txt') as fin:
    test_names = [line.split()[0] for s in fin.readlines()]

