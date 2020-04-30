python_version=3.6
source /opt/intel/openvino/bin/setupvars.sh
python3.6 test.py -o ../../images -m ../../model/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml -i ../../images/people-counter-image.png -r a
