# Important otherwise openvino might try to use latest installed python version.
# INPUT="CAM"
#INPUT="CAM"
INPUT="resources/Pedestrian_Detect_2_1_1.mp4"
SHOWINFO=true
MESSAGE="Udacity"
#INPUT="resources/Pedistrain_Hong_Kong.mp4"
python_version=3.5
source /opt/intel/openvino/bin/setupvars.sh
# To download model use following commands
# cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
# sudo ./downloader.py --name vehicle-attributes-recognition-barrier-0039 --precisions INT8 -o /home/workspace

MODEL="model/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml"
echo "*************"
echo "INPUT: " $INPUT
echo "MODEL: " $MODEL
echo "*************"

LIB="/opt/intel/openvino_2020.2.120/deployment_tools/inference_engine/lib/intel64/libclDNNPlugin.so"

# -t $TAG -c $LIB
#python3.5 pipeline/main.py -i $INPUT -m $MODEL -p 0.7
python3.5 pipeline/main.py -si $SHOWINFO --message $MESSAGE -i $INPUT -m $MODEL -pt 0.5 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
