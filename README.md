# YOLO_V3 Inference With Our Trained Model

**Table Of Contents**
- [How does this sample work?](#how-does-this-sample-work)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)


## How does this sample work?

This example uses a pre-trained YoloV3 model for which we have the .weights and .cfg files. These are converted to the Open Neural Network Exchange (ONNX) format in `conv_weights2onnx.py` (only has to be done once).

This YOLOv3 ONNX representation is used to perform inferences on both images and video. The predicted bounding boxes are finally drawn to the original input and saved to disk.



**Note:** This sample is not supported on Ubuntu 14.04 and older.

## Prerequisites

For specific software versions, see the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html).

1.  Install the dependencies for Python.
    ```sh
    python3 -m pip install -r requirements.txt
    ```

## Running the sample

1.  Create an ONNX version of YOLOv3 with the following command.
    ```sh
    python3 conv_weights2onnx.py <path/to/cfg> <path/to/namesfile> <path/to/weight> <batch_size>
    ```
    
2.  Run inference with the generated ONNX file on a sample image
    ```sh
    python3 image_inference.py <path/to/onnx-file> <path/to/names> <path/to/image> <batch_size>
    ```
    For example:
    ```sh
    python3 image_inference.py yolov3.onnx signal.names train.jpg 1
    ```


3.  Run inference with the generated ONNX file on a sample image
    ```sh
    python3 video_inference.py <path/to/onnx-file> <path/to/names> <path/to/video> <batch_size> <show>
    ```
    For example:
    ```sh
    python3 video_inference.py yolov3.onnx signal.names video.mp4 1 show
    ```


4.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
    ```
    Inference with ONNX ....
    The model expects input shape:  [1, 3, 320, 320]
    OpenCV: FFMPEG: tag 0x47504a4d/'MJPG' is not supported with codec id 7 and format 'mp4 / MP4 (MPEG-4 Part 14)'
    OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'
    Shape of the network input:  (1, 3, 320, 320)
    Shape of the network input:  (1, 3, 320, 320)
    Shape of the network input:  (1, 3, 320, 320)
    -----------------------------------
       max and argmax : 0.000407
                  nms : 0.000131
    Post processing total : 0.000538
    -----------------------------------
    ```


