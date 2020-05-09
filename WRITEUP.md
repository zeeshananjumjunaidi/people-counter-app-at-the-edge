# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The process behind converting custom layers involves...

Ans: 
  For Tensorflow:
    Register custom layer as extension in model optimizer OR
    You need some sub graph that shoud not be in IR and also have another subgraph for that operation. Model Optimizer provides such solution as "Sub Graph Replacement" OR
    Pass the custom operation to Tensorflow to handle during inference.


Some of the potential reasons for handling custom layers are...

  Ans:
    The Custom layes known as per their name "Custom" means modified or new. There are variety of frameworks which are used for training the deep learning models such as Keras, Tensorflow, ONNX, Caffe etc.

    All these frameworks have their own methods to process the tensors (Data) so it may possible that some functions are not available or behaves diffrently in each other.

    Hence Custom Layer support neccessary for Model Optimizer so that the unsupported operations can be supported through dependent framework during runtime inference.

    Model Optimizer query each layer of trained model from the list of known layers (Supported layers) before building the model's internal representation. It also optimizes the model by following three steps. Quantization, Freezing and Fusing. At last it generated the intermidiate representation from the trained model.
---
## Comparing Model Performance
file: ```pipeline/compare_model.py```
My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...
pre-model size was 208 MB and post-conversion size is 67.6 MB
The inference time of the model pre- and post-conversion was...

PERFORMANCE OF THE ORIGINAL MODEL 
took '2118.613ms / ~2s' time for inference with 98.036915% accuracy.
OPENVINO PERFORMANCE AFTER CONVERTING THE MODEL
took '13.977ms' time for ineference and  98.36082% accuracy!
As you can see there is huge difference time as original model took approx 2 seconds and optimized model to 0.13 second.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

    Crowd Detection
    Security Systems
    Audience Statistics
    Queue counter
    Social distance checking

Each of these use cases would be useful because...

**Crowd Detection** It is very useful to know if there a smooth flow of crowd in mall, sidewalk  or any other place
 and also to know if there is gathering / protest/ riot etc... In this way system could alert the authorities to hanlde the situation.
 
 **Security Systems** This can be useful for detection of the number of people, or if there is someone in restricted area.
 
 **Audience Statistics** Presentor, or Event Manager can easily count the number of attendees in any type of event to make proper arrangement.
 
 **Queue counter** This system is very useful for checkout counter to adjust the queue, and fully utilize the resources.
 
 **Social distance** Limiting face-to-face contact with others is the best way to reduce the spread of coronavirus disease 2019 (COVID-19). People detection can easily be used by as detecting distance b/w persons. 
 ref: https://www.cdc.gov/coronavirus/2019-ncov/prevent-getting-sick/social-distancing.html

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...
  - False positive has been detected in some cases, such as pedistrain image, or a small tree look far from the camera.
  - Low luminosity, occluded by big objects or by other person cause false negative results.
  - Large differences in training dataset camera angle with the inference images would also result in invalid result.
