Install all prerequisite for all frameworks, openvino toolkit supports
run 
``` bash 
/opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites.sh
```
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