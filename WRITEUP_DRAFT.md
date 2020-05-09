# Deploy a People Counter App at the Edge

| Details           |               |
|-----------------------|---------------|
| Programming Language: |  Python 3.5 or 3.6 |
| Libraries, Framework & Tools: |  NodeJS, TensorFlow, OpenVino, OpenCV |
| Auhtor: |  Zeeshan Anjum Junaidi |


## Installation
Download Latest openvino toolkit.

Install all prerequisite for all frameworks, openvino toolkit supports
run 
``` bash 
/opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites.sh
```


## Overview
The people counter application will demonstrate how to create a smart video IoT solution using Intel® hardware and software tools. The app will detect people in a designated area, providing the number of people in the frame, average duration of people in frame, and total count.
In this project I used **Pedestrain** 🚶🏻‍♀️🚶🏻‍♂️ detection model from **...** and optimize it using **OpenVino 2020.2+** library provided by **Intel®**. This project contains different modules that are integrated together as a system.

## Model Conversion

I used ssd_mobilenet_v2_coco model from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
You can download compress model files from http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz and extact it.

Here's what I used to convert the SSD MobileNet V2 model from TensorFlow model to Intermediate Representation:
I put the downloaded model inside **model/source/ssd_mobilenet_v2_coco_2018_03_29** project folder. (Consider relative path for the model and absolute path for the openvino files)

```
python3.5 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model model/source/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config model/source/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```
then **Model Optimizer** mo.py generates Intermediate Representation (IR), model/output/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.bin and 
model/output/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml
.xml file represents model architecture and other important meta data.
.bin (binary) contains model weights and biases.


----

### Pipeline
Location ```./pipeline```
This folder contains all the required files used for inference, from loading model, communicate with NodeJS server using MQTT, stream media to FFMPEG server.

To run pipeline navigate to project root folder and run 
```./run.sh```

 
### Web Service
Location ```./webservice/```
- #### UI
  This module is responsible for user interface. This web app will communicate with NodeJS server over MQTT.
  This module also display video streaming, allow user to play/pause streaming.

To run ui app navigate to webservice/ui then run
```npm run dev```

- #### Server
  Server will handle communication b/w web UI and the core pipeline.
  
To run web server navigate to webservice/server/node-server then run
```node ./server.js```

### FFMPEG
Location ```./ffmpeg/ ```
    This will only contain configuration to run ffserver for streaming.
  
To run ffmpeg server navigate to project root then run
```sudo ffserver -f ./ffmpeg/server.conf```
  
----


Following are the optimization methods for the model, provided by the openvino toolkit.

**Quantization**

Quantization is how many bits are used to represent the weights and biases of the model. Quantization is the process of reducing the precision of a model.

With the OpenVINO™ Toolkit, models usually default to FP32, or 32-bit floating point values, while FP16 and INT8, for 16-bit floating point and 8-bit integer values, are also available (INT8 is only currently available in the Pre-Trained Models; the Model Optimizer does not currently support that level of precision). FP16 and INT8 will lose some accuracy, but the model will be smaller in memory and compute times faster. Therefore, quantization is a common method used for running models at the edge.

**Freezing**

Freezing in this context is used for TensorFlow models. Freezing TensorFlow models will remove certain operations and metadata only needed for training, such as those related to backpropagation. Freezing a TensorFlow model is usually a good idea whether before performing direct inference or converting with the Model Optimizer.

**Fusion**

Fusion relates to combining multiple layer operations into a single operation. For example, a batch normalization layer, activation layer, and convolutional layer could be combined into a single operation. This can be particularly useful for GPU inference, where the separate operations may occur on separate GPU kernels, while a fused operation occurs on one kernel, thereby incurring less overhead in switching from one kernel to the next.

**Intermediate Representations** 
(IRs) are the OpenVINO™ Toolkit’s standard structure and naming for neural network architectures. A Conv2D layer in TensorFlow, Convolution layer in Caffe, or Conv layer in ONNX are all converted into a Convolution layer in an IR.

The IR is able to be loaded directly into the Inference Engine, and is actually made of two output files from the Model Optimizer: an XML file and a binary file. The XML file holds the model architecture and other important metadata, while the binary file holds weights and biases in a binary format. 

Command to generate IR 
export MOD_OPT=/opt/intel/openvino/deployment_tools/model_optimizer

python3.5 $MOD_OPT/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config $MOD_OPT/extensions/front/tf/ssd_v2_support.json

Cutting TF Model

# Project Workflow
## Convert a Model into an Intermediate Representation with the Model Optimizer
## Load the Model Intermediate Representation into the Inference Engine
## Check for Custom Layers
## Handle Inference Requests Asynchronously
# Return Results

**Reference**
https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html
https://classroom.udacity.com/nanodegrees/nd131/

**Pipeline reference**
https://github.com/intel-iot-devkit/people-counter-python

**Object detection zoo**
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

**Learning Resources**
https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_Model_Optimization_Techniques.html

**CREDIT**
Pedistrain video for testing
https://www.youtube.com/watch?v=dju1olF6ilM
