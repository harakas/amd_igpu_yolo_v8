# Based on
#   https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/pytorch-install.html#using-docker-with-pytorch-pre-installed
#   https://pytorch.org/hub/ultralytics_yolov5/

FROM rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1

RUN apt-get -y update
RUN apt-get -y upgrade

RUN pip install -U pip

RUN pip install -U ultralytics

RUN pip install -U 'gitpython>=3.1.30'
RUN pip install -U 'Pillow>=10.0.1'
RUN pip install -U 'numpy>=1.23.5'
RUN pip install -U 'scipy>=1.11.4'
RUN pip install -U 'onnx>=1.12.0'
RUN pip install -U onnxruntime

RUN apt-get install -y migraphx

RUN mkdir /opt/cwd

WORKDIR /opt/cwd

ENTRYPOINT ["/opt/conda/envs/py_3.10/bin/python"]

