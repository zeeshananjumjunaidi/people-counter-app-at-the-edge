***Error #1:***
``` diff
- File "/opt/intel/openvino_2020.2.120/python/python3.6/openvino/inference_engine/__init__.py", line 1, in <module>
-    from .ie_api import *
- ImportError: libpython3.6m.so.1.0: cannot open shared object file: No such file or directory

```
***solution:***
```    
sudo add-apt-repository ppa:fkrull/deadsnakes
sudo apt update
sudo apt install libpython3.6-dev
```
To Install sklearn for python3.5 on linux OS.
run following
```
python3.5 -m pip install scikit-learn
==0.21
```
latest versions aren't supported