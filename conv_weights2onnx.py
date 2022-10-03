import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import onnxruntime

from tool.utils import *
from tool.darknet2onnx import *


def main(cfg_file, namesfile, weight_file, image_path, batch_size):
    if batch_size <= 0:
        onnx_path_demo = transform_to_onnx(cfg_file, weight_file, batch_size)
    else:
        # Transform to onnx as specified batch size
        transform_to_onnx(cfg_file, weight_file, batch_size)
        # Transform to onnx as demo
        onnx_path_demo = transform_to_onnx(cfg_file, weight_file, 1)

if __name__ == '__main__':
    print("Converting to onnx ....")
    if len(sys.argv) == 6:
        cfg_file = sys.argv[1]
        namesfile = sys.argv[2]
        weight_file = sys.argv[3]
        image_path = sys.argv[4]
        batch_size = int(sys.argv[5])
        main(cfg_file, namesfile, weight_file, image_path, batch_size)
    else:
        print('Please run this way:\n')
        print('  python conv_weights2onnx.py <cfgFile> <namesFile> <weightFile> <imageFile> <batchSize>')

