# Important otherwise openvino might try to use latest installed python version.
INPUT="$1"
python_version=3.5
source /opt/intel/openvino/bin/setupvars.sh
# To download model use following commands
# cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
# sudo ./downloader.py --name vehicle-attributes-recognition-barrier-0039 --precisions INT8 -o /home/workspace

MODEL="intel/human-pose-estimation-0001/FP32/human-pose-estimation-0001.xml"
echo "*************"
echo "INPUT: " $INPUT
echo "MODEL: " $MODEL
echo "*************"

LIB="/opt/intel/openvino_2020.2.120/deployment_tools/inference_engine/lib/intel64/libclDNNPlugin.so"

# -t $TAG -c $LIB
python3.5 ./main.py -i $INPUT -m $MODEL 
