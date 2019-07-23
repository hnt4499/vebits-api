# vebits_api
A high-level, comprehensive package that leverages user's experience when working with Tensorflow's Object Detection API.

## Overview
This package has been developed to turn my works at [Vebits](https://vebits.com/en) into a friendly, easy-to-use API that facilitate user's experience when working with [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). New features are being developed and tested to working with [DarkNet](https://github.com/pjreddie/darknet)/[Darkflow](https://github.com/thtrieu/darkflow) for training YOLO models running real-time on mobile devices; as well as with [MMdetection](https://github.com/open-mmlab/mmdetection) for high-performance, highly scalable object detection toolbox.
## Dependencies
All dependencies are listed under `requirement.txt`
```
certifi==2019.6.16
cycler==0.10.0
decorator==4.4.0
imageio==2.5.0
imgaug==0.2.9
imutils==0.5.2
kiwisolver==1.1.0
matplotlib==3.1.1
networkx==2.3
numpy==1.16.4
opencv-python==4.1.0.25
pandas==0.25.0
Pillow==6.1.0
protobuf==3.9.0
pyparsing==2.4.1
python-dateutil==2.8.0
pytz==2019.1
PyWavelets==1.0.3
scikit-image==0.15.0
scipy==1.3.0
Shapely==1.6.4.post2
six==1.12.0
tqdm==4.32.2
```
Optionally, the following packages are required for the API to work seamlessly with 

 - Tensorflow's Object Detection API: `tensorflow`
 ```
 pip install tensorflow-gpu
 ```
- Darknet/Darkflow for YOLO models: `darkflow`
```
git clone https://github.com/thtrieu/darkflow.git
cd darkflow
pip install -e .
```
- MMdetection toolbox: details on installation can be found [here](https://github.com/open-mmlab/mmdetection/blob/master/INSTALL.md).

## Installation
To install the latest stable release of this package, simply run:
```
pip install vebits_api
```
Alternatively, to build the project from source in development mode and allow the changes to take effect immediately:
```
git clone https://github.com/hnt4499/vebits_api/
cd vebits_api/
pip install -e .
```
or
```
pip install git+https://github.com/hnt4499/vebits_api.git
```
That's it! To make use of available scripts for data manipulating/processing/visualization, simply copy all scripts under `scripts` folder to your working directory.

## TODO:
- [ ] Complete README.md: Requirements, Build from source, Usage, Reference, Examples.
- [ ] Incorporate DarkNet into this package.
- [ ] 
<!--stackedit_data:
eyJoaXN0b3J5IjpbNzcxNjc2MDk0LDEzNjAxODU4LDIxNDQ4NT
g3XX0=
-->