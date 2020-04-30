import argparse
import os
from openvino.inference_engine import IENetwork, IECore
from openvino_helper import load_to_IE, preprocessing
import cv2
import timeit
from matplotlib import pyplot as plt
import time
from utility import *
# CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libMKLDNNPlugin.so"

class SimpleInference:
    def __init__():
        self.model_bin = None
        self.input_blob = None
        self.out_blob = None
        self.input_shape = None
        self.infer_request_handle = None
        self.exec_net = None

    def load_IE(model_xml):
        ### Load the Inference Engine API
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # ### Add a CPU extension, if applicable.
        # # plugin.add_extension(CPU_EXTENSION, "CPU")

        # ### Get the supported layers of the network
        # supported_layers = plugin.query_network(network=net, device_name="CPU")

        # ### Check for any unsupported layers, and let the user
        # ### know if anything is missing. Exit the program, if so.
        # unsupported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        # if len(unsupported_layers) != 0:
        #     print("Unsupported layers found: {}".format(unsupported_layers))
        #     print("Check whether extensions are available to add to IECore.")
        #     exit(1)

        ### Load the network into the Inference Engine
        # plugin.load_network(net, "CPU")
        
        # model = IENetwork(model_xml, model_bin)
        
        # Loading model from openvino helper

        exec_net, input_shape = load_to_IE(model_xml,False)
        self.input_shape = input_shape
        self.exec_net = exec_net
        print("IR successfully loaded into Inference Engine.")
        return exec_net, input_shape 

    def async_inference(exec_net, cur_request_id, image):
        ### TODO: Add code to perform asynchronous inference
        ### Note: Return the exec_net
        exec_net.start_async(request_id=cur_request_id, inputs={self.input_blob: image})
        while True:
            status = exec_net.requests[0].wait(-1)
            if status == 0:
                break
            else:
                time.sleep(1)
        return exec_net


    def sync_inference(exec_net, input_blob, image):
        ### TODO: Add code to perform synchronous inference
        ### Note: Return the result of inference
        result = exec_net.infer({input_blob: image})

    def perform_inference(exec_net, request_type, input_image, input_shape):
        '''
        Performs inference on an input image, given an ExecutableNetwork
        '''
        # Get input image
        image = cv2.imread(input_image)
        # Extract the input shape
        n, c, h, w = input_shape
        # Preprocess it (applies for the IRs from the Pre-Trained Models lesson)
        preprocessed_image = preprocessing(image, h, w)

        # Get the input blob for the inference request
        input_blob = next(iter(exec_net.inputs))

        # Perform either synchronous or asynchronous inference
        request_type = request_type.lower()
        if request_type == 'a':
            print('async inference')
            output = async_inference(exec_net, input_blob, preprocessed_image)
        elif request_type == 's':
            print('sync inference')
            output = sync_inference(exec_net, input_blob, preprocessed_image)
        else:
            print("Unknown inference request type, should be 'A' or 'S'.")
            exit(1)

        # Return the exec_net for testing purposes
        return output

    def main():
        args = get_args()
        exec_net, input_shape = load_IE(args.m)
        cur_request_id=0
        inf_start = time.time()
        infer_network = perform_inference(exec_net, args.r, args.i, input_shape)
        if infer_network.wait(cur_request_id) == 0:
                det_time = time.time() - inf_start
                # Results of the output layer of the network
                res = infer_network.infer().get("detection_out")
                print(res.shape)
                print("infer result: label:%f confidence:%f left:%f top:%f right:%f bottom:%f" %(res[0][0][0][1], res[0][0][0][2], res[0][0][0][3], res[0][0][0][4], res[0][0][0][5], res[0][0][0][6]))
                image = cv2.imread(args.i)
                #image = cv2.imread(input_image)
                output_img,person_counts = get_draw_boxes_on_image(res,image)
                print(output_img.shape)
                print(person_counts)
                file_name = os.path.join(args.o,'output.png')
                print(file_name)
                cv2.imwrite(file_name, output_img) 
                # print('output image shape:{}'.format(output_img.shape))
                # plt.plot(output_img)
                # plt.show()
                # print(infer_network.outputs)
                # result = infer_network.get_output(cur_request_id)
                if args.perf_counts:
                    perf_count = infer_network.performance_counter(cur_request_id)
                    performance_counts(perf_count)
            # print(infer_network)


