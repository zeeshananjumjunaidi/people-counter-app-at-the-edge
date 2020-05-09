#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore
from openvino_helper import load_to_IE

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.net = None
        self.plugin = None
        self.input_blob = None
        self.out_blob = None
        self.net_plugin = None
        self.infer_request_handle = None
        # my own parameters
        self.input_shape = None
        self.exec_net = None

    def load_model(self, model, device, input_size, output_size, num_requests, cpu_extension=None, plugin=None):
        ### TODO: Load the model ###
        
        self.plugin = IECore()
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        self.net = self.plugin.read_network(model=model_xml, weights=model_bin)
        self.net_plugin = self.plugin.load_network(self.net,device_name="CPU")

        ### TODO: Check for supported layers ###
        supported_layers = self.plugin.query_network(network=self.net, device_name="CPU")
        
        ### Note: You may need to update the function parameters. ###
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)
        ### TODO: Add any necessary extensions ###        
        if cpu_extension:
            self.plugin.add_extension(cpu_extension,"CPU")
        
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))

        # return exec_net, input_shape 
        return self.plugin, self.get_input_shape()

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        # return self.input_shape
        return self.net.inputs[self.input_blob].shape

    def async_inference(self,exec_net, request_id, image):
        ### TODO: Add code to perform asynchronous inference
        ### Note: Return the exec_net
        self.infer_request_handle = self.net_plugin.start_async(request_id=request_id, inputs={self.input_blob: image})
        return exec_net

    def exec_network(self,id,image):
        ### TODO: Start an asynchronous request ###
        self.infer_request_handle = self.net_plugin.start_async(request_id=id, inputs={self.input_blob: image})
        ### TODO: Return any necessary information ###

        ### Note: You may need to update the function parameters. ###
        return self.net_plugin

    def wait(self,id):
        ### TODO: Wait for the request to be complete. ###
        status = self.net_plugin.requests[id].wait(-1)
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return status

    def get_output(self,request_id, output=None):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        if output:
            res = self.infer_request_handle.outputs[output]
        else:
            res = self.net_plugin.requests[request_id].outputs[self.out_blob]
        return res

    def dispose(self):
        """
        Remove all the instances
        :return: None
        """
        del self.net_plugin
        del self.plugin
        del self.net