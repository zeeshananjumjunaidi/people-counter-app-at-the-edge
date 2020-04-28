for error:
File "/opt/intel/openvino_2020.2.120/python/python3.6/openvino/inference_engine/__init__.py", line 1, in <module>
    from .ie_api import *
ImportError: libpython3.6m.so.1.0: cannot open shared object file: No such file or directory
solution:
sudo add-apt-repository ppa:fkrull/deadsnakes
sudo apt update
sudo apt install libpython3.6-dev
for error while installing matplotlib:
      File "/tmp/pip-build-1__czfja/matplotlib/setup.py", line 139
        raise IOError(f"Failed to download jquery-ui.  Please download "
solution:
download jquery

if no package available for unzip then use following
sudo apt-get install unzip

unzip jquery
unzip jquery-ui-1.12.1.zip
