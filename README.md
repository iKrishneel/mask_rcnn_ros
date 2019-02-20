
# Mask RCNN Object Detector
---
A `ROS` Node for detecting objects using [Mask RCNN](https://github.com/matterport/Mask_RCNN). The detector 
is build on the Tensorflow and Keras ecosystem.  

## Requirements
---
>Install the following dependencies before compiling this package
- [ROS Kinetic or higher](http://wiki.ros.org/kinetic)
- [CUDA (8.0 or higher)](https://developer.nvidia.com/cuda-downloads)
- [OpenCV 3.0 or higher](https://github.com/opencv/opencv)
- [Python 3 or higher](https://www.python.org/download/releases/3.0/)
- [TensorFlow 1.8.0 or higher](https://www.tensorflow.org/)
- [Keras 2.1.6](https://keras.io/)

Install the required deps on the `virtualenv`

## Installation
----
>It is better to use the python *virtual environment* to install the detector dependencies and note that it only works on `Python 3`

>Install python Virtual Environment
```bash
$ sudo pip install virtualenv
$ sudo pip install virtualenvwrapper
```

>Creating Virtual Environment
```bash
$ venv_name=mrcnn
$ mkvirtualenv --python=python3 $venv_name
```

>Install the dependencies
```bash
$ pip install -r requirements.txt
```

## Downloading the Package
---
> Clone the package to the ROS workspace using git tools
```bash
$ git clone https://github.com/iKrishneel/mask_rcnn_ros.git
$ cd jsk_hsr_wrs
$ git pull --all
$ git submodule update --init
```

## Compilation
------------
> The package is a ROS node and can be complied using the catkin tools
```bash
  $ catkin build jsk_hsr_wrs
  $ source $HOME/.bashrc
```

## Running
---
>Source to activate the `virtualenv` containing the installed deps.

1. Running the node as a publisher
```bash
roslaunch object_detector object_detector.launch image:=<image_topic_name> is_service:=false debug:=true
```
2. Running the node as a service
```bash
roslaunch object_detector object_detector.launch image:=<image_topic_name> is_service:=true debug:=false
```

## Arguments
---
The following arguments can be set on the `roslaunch` above.
- `image`: Image topic name
- `is_service`: boolean flag when set launches the node as a ros service.
- `debug`: boolean flag when set shows the detection results in opencv visualization `cv.imshow()`
- `detection_threshold`: threshold to filter the detection results [0, 1]
- `model`: path to the training keras model file. The model is with `.h5` extension.
- `class_labels`: text file containing the mapping of the class label with the class name

## Training
---
The Mask RCNN can be trained as follows on custom dataset.

##### Dataset Loading
Modify the [dataloader script](https://github.com/iKrishneel/mask_rcnn_ros/blob/master/scripts/train/custom_dataloader.py) based on your dataset format or use the COCO dataloader. The current dataset loader in this package is different from the COCO loader that comes with the Mask RCNN. It is useful when training Mask RCNN on custom data. 
Current loader assumes that the dataset directory contains two folders and a textfile:
- `image/` which contains all the images (eg. `image_0001.jpg`)
- `label` contains folders (named after the image name eg. `image_0001/` corresponding to each image. Inside the subfolder `image_0001` are mask images of target objects for learning (eg `1.png, 4.png`. The name of the mask image in the subfolder corresponds to the **class label** of the objects)
- `class.txt` - textfile contains the class names corresponding to each label on a new line(eg: apple 1 [separated by single white space])

##### Hyperparameter Tunning
Before starting the training, you will need to tune some hyperparameters.

1. `NUM_CLASSES` [#L53](https://github.com/iKrishneel/mask_rcnn_ros/blob/master/scripts/train/mask_rcnn_trainer.py#L53) in the dataset
2. `Learning Params` [#44](https://github.com/iKrishneel/mask_rcnn_ros/blob/master/scripts/train/mask_rcnn_trainer.py#L144) set learning parameters like `epochs`, `layers` [which layers of ReNet to train] and `LEARNING_RATE` (if necessary)

##### Training Command

```bash
$ python mask_rcnn_trainer.py --dataset <path_to_dataset> --model <path_to_model_used_for_initialization>  train
````
