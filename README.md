# ROCm pytorch yolov5 example
A simple example on how to run the [ultralytics/yolov5](https://pytorch.org/hub/ultralytics_yolov5/) inference model on the AMD ROCm platform with pytorch.

I have an [ASRock 4x4 BOX-5400U mini computer](https://www.asrockind.com/en-gb/4X4%20BOX-5400U) with integrated AMD graphics. It has a GPU that is capable of running neural networks/pytorch. This here is an example/description on how to get it working.

Here we:
 * create a Docker image named `rocm-pytorch` that contains the ROCm and pytorch software environment
 * modify command line script `rocm_python` that runs this Docker image inline as a `python` wrapper
 * use this script to run the yolo5.py example script for inference on [wolf.jpg](wolf.jpg)

## Description of environment variables specified in the [rocm_python](rocm_python) script

AMD mini computers/laptops with integrated GPUs do not run out of the box but need special environment variables to work properly.

### `HSA_ENABLE_SDMA=0`

Without this the GPU memory transfers will silently hang/fail. Based on various internet forums it is because at this moment no consumer AMD motherboards with integrated GPUs support PCIe atomics. They are apparently needed for DMA memory transfers between the GPU and CPU. Why did AMD make the default setting for consumer level GPU-s to silently hang/crash? You need to ask them.

### `HSA_OVERRIDE_GFX_VERSION=9.0.0`

ROCm comes with precompiled/preoptimized kernel files and settings for some of their GPUs, but not for the one I have (gfx90c). To get this GPU working we need to fake the GPU version so that ROCm loads the more generic (but less optimized) kernels and settings instead. It seems to work, at least for my specific GPU (gfx90c).

You need to either delete this if yours is already supported or specify a version suitable for your GPU. You can find your current GPU name by running `rocminfo`:

```
$ rocminfo
...
*******
Agent 2
*******
  Name:                    gfx90c
  Uuid:                    GPU-XX
  Marketing Name:          AMD Radeon Graphics
...
```

See https://github.com/ROCm/ROCm/issues/1743#issuecomment-1149902796 for more examples for other GPUs.

## Build a Docker image

As installing all the software is quite involved and there are many versions that conflict we resort to using containers instead. We build a Docker image named `rocm-pytorch` by running:

```
$ docker build -t rocm-pytorch .
```

See [Dockerfile](Dockerfile) for details. It is built on top of the prebuilt AMD/ROCm provided pytorch Docker image.


## Run the example script:

As input file we use an image of a wolf:

![wolf pup](wolf.jpg)

We run the [yolo5.py](yolo5.py) script that runs inference on this image.

```
$ ./rocm_python yolo5.py
torch.cuda.is_available() True
torch.cuda.device_count() 1
torch.cuda.current_device() 0
torch.cuda.get_device_name(0) AMD Radeon Graphics
/opt/conda/envs/py_3.9/lib/python3.9/site-packages/torch/hub.py:294: UserWarning: You are about to download and run code from an untrusted repository. In a future release, this won't be allowed. To add the repository to your trusted list, change the command to {calling_fn}(..., trust_repo=False) and a command prompt will appear asking for an explicit confirmation of trust, or load(..., trust_repo=True), which will assume that the prompt is to be answered with 'yes'. You can also use load(..., trust_repo='check') which will only prompt for confirmation if the repo is not already trusted. This will eventually be the default behaviour
  warnings.warn(
Downloading: "https://github.com/ultralytics/yolov5/zipball/master" to /var/lib/jenkins/.cache/torch/hub/master.zip
YOLOv5 🚀 2024-1-19 Python-3.9.18 torch-2.2.0a0+gitd925d94 CUDA:0 (AMD Radeon Graphics, 512MiB)

Downloading https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt to yolov5n.pt...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3.87M/3.87M [00:01<00:00, 3.23MB/s]

Fusing layers...
YOLOv5n summary: 213 layers, 1867405 parameters, 0 gradients, 4.5 GFLOPs
Adding AutoShape...
image 1/1: 768x1152 1 dog
Speed: 7.7ms pre-process, 395.3ms inference, 134.9ms NMS per image at shape (1, 3, 448, 640)
Saved 1 image to runs/detect/exp
Benchmark: inference min 0.019536256790161133, median 0.020015954971313477
```

If you see what I see, it works. An illustrated image with detections will be generated under [./runs/detect/](./runs/detect/):

![inference result](runs/detect/exp/wolf.jpg)

I hope this example was of help to you. Good luck!
