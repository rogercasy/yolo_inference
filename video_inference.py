import time
import sys
import onnx
import os
import argparse
import random
import numpy as np
import cv2
import onnxruntime
from threading import Thread, enumerate
from queue import Queue


from tool.utils import *
from tool.darknet2onnx import *

def print_detections(detections, coordinates=False):
    print("\nObjects:")
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        if coordinates:
            print("{}: {}%    (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(label, confidence, x, y, w, h))
        else:
            print("{}: {}%".format(label, confidence))

def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted

def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height
    
def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video

def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path
    
def draw_boxes(detections, image, colors):
    import cv2
    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
    return image

def inference(darknet_image_queue, detections_queue, fps_queue):
    while cap.isOpened():
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()        
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: darknet_image})
        detections = post_processing(darknet_image, 0.4, 0.6, outputs)
        #class_names = load_class_names(namesfile)        
        detections_queue.put(detections)
        fps = int(1/(time.time() - prev_time))
        fps_queue.put(fps)
        print("FPS: {}".format(fps))
    cap.release()
 
def video_capture(frame_queue, darknet_image_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)
        #print("FRAME.....", frame)                      
        frame_queue.put(frame)        
        frame_resized = np.transpose(frame_resized, (2, 0, 1)).astype(np.float32)
        frame_resized = np.expand_dims(frame_resized, axis=0)
        frame_resized /= 255.0
        print("Shape of the network input: ", frame_resized.shape)
        img_for_detect = frame_resized        
        darknet_image_queue.put(img_for_detect)        
    cap.release()

def drawing(frame_queue, detections_queue, fps_queue):
    random.seed(3)  # deterministic bbox colors
    video = set_saved_video(cap, 'output.mp4', (video_width, video_height))
    while cap.isOpened():
        frame = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        detections_adjusted = []
        if frame is not None:
            print(detections)
            image = frame
            if detections != [[]]:
                for bbox in detections:
                    image = plot_boxes_cv2(frame, bbox, class_names=class_names)
                #for bbox in detections[0]:
                #    bbox_adjusted = convert2original(frame, bbox)
                #    detections_adjusted.append(('signal', confidence, bbox_adjusted))            
            #image = draw_boxes(detections_adjusted, frame, 'red')
            video.write(image)
            if show != 'dont_show':
                cv2.imshow('Inference', image)                
            if cv2.waitKey(fps) == 27:
                break
    cap.release()
    video.release()
    cv2.destroyAllWindows()
    


if __name__ == '__main__':
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)
    
    print("Inference with ONNX ....")
    if len(sys.argv) == 6:
        onnx_path = sys.argv[1]
        namesfile = sys.argv[2]
        video_path = sys.argv[3]
        batch_size = int(sys.argv[4])
        show = sys.argv[5]
        session = onnxruntime.InferenceSession(onnx_path)
        print("The model expects input shape: ", session.get_inputs()[0].shape)
        input_path = str2int(video_path)
        cap = cv2.VideoCapture(input_path)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))       
        darknet_height = session.get_inputs()[0].shape[2]
        darknet_width = session.get_inputs()[0].shape[3]
        class_names = load_class_names(namesfile)     
        
        Thread(target=video_capture, args=(frame_queue, darknet_image_queue)).start()    
        Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue)).start()
        Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue)).start()    
        
    else:
        print('Please run this way:\n')
        print('  python inference_onnx.py <onnxFile> <namesFile> <imageFile> <batchSize> <show>')

