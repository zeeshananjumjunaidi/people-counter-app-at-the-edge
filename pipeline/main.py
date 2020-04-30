import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
from openvino_helper import preprocessing
from utility import *
# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
FONT = cv2.FONT_HERSHEY_SIMPLEX 

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = None

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    cur_request_id = 0
    last_count = 0
    total_count = 0
    start_time = 0
    if args.input == 'CAM':
        input_stream = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image_mode = True
        input_stream = args.input
    # Checks for video file
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"
    ### TODO: Load the model through `infer_network` ###
    n, c, h,w = infer_network.load_model(args.model, args.device, 1, 1,
                                          cur_request_id, args.cpu_extension)[1]
    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(input_stream)
    if input_stream:
        cap.open(args.input)
    if not cap.isOpened():
        log.error("ERROR! Unable to open video source")

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        ### TODO: Pre-process the image as needed ###
        image = preprocessing(frame,h,w)
        
        ### TODO: Start asynchronous inference for specified request ###
        inf_start = time.time()
        #infer_network.async_inference(infer_network.exec_net,cur_request_id, image)
        infer_network.exec_network(cur_request_id, image)
        ### TODO: Wait for the result ###
        if infer_network.wait(cur_request_id) == 0:
            ### TODO: Get the results of the inference request ###
            det_time = time.time() - inf_start
            result = infer_network.get_output(cur_request_id)
            # print(result.shape)
            # print(type(result))
            # res = infer_network.infer().get("detection_out")
            ### TODO: Extract any desired stats from the results ###

            output_img,person_counts = get_draw_boxes_on_image(result,frame,prob_threshold)
            if(person_counts>0):
                print(person_counts)
            cv2.putText(frame,  
                'Persons found {}'.format(person_counts),  
                (50, 50),  
                FONT, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4) 
            cv2.imshow('frame',output_img)
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

        ### TODO: Send the frame to the FFMPEG server ###

        ### TODO: Write an output image if `single_image_mode` ###
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    infer_network.clean()


def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()