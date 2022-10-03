import time
import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import onnxruntime

from tool.utils import *
from tool.darknet2onnx import *



def main(onnx_path, namesfile, image_path, batch_size):
    session = onnxruntime.InferenceSession(onnx_path)
    print("The model expects input shape: ", session.get_inputs()[0].shape)
    image_src = cv2.imread(image_path)
    t0 = time.time()
    detect(session, image_src, namesfile)    
    print("Predict Time: %ss" % (time.time() - t0))


def detect(session, image_src, namesfile):
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]

    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H),    interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    print("Shape of the network input: ", img_in.shape)

    # Compute
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: img_in})

    boxes = post_processing(img_in, 0.4, 0.6, outputs)

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(image_src, boxes[0], savename='predictions_onnx.jpg', class_names=class_names)

if __name__ == '__main__':
    print("Inference with ONNX ....")
    if len(sys.argv) == 5:
        onnx_path = sys.argv[1]
        namesfile = sys.argv[2]
        image_path = sys.argv[3]
        batch_size = int(sys.argv[4])
        main(onnx_path, namesfile, image_path, batch_size)
    else:
        print('Please run this way:\n')
        print('  python inference_onnx.py <onnxFile> <namesFile> <imageFile> <batchSize>')

