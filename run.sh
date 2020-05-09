# Important otherwise openvino might try to use latest installed python version.
python_version=3.5

# 640x480 for camera
#INPUT="CAM"
X=854
Y=480
# 854x480 for traffic.mp4
INPUT="resources/traffic.mp4"
# To render extra info overlay on top of output image frame.
SHOWINFO=true
# MESSAGE text will appear on top most of the screen.
MESSAGE="[Udacity]"
# Threshold for the detection confidence.
THRESHOLD=0.5
# Sourcing openvino
source /opt/intel/openvino/bin/setupvars.sh
# location of model .xml file, keep the .bin (weight & biases) file in the same directory 
# as app'll try to load it from the same directory.
MODEL="model/output/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml"
# print output
echo "*************"
echo "INPUT: " $INPUT
echo "MODEL: " $MODEL
echo "*************"

python3.5 pipeline/main.py -si $SHOWINFO --message $MESSAGE -i $INPUT -m $MODEL -pt $THRESHOLD   | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size "$X"x$Y -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
