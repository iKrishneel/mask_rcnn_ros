#!/usr/bin/env python

import os
import sys
import time
import numpy as np
import imgaug
import urllib.request

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mask_rcnn')
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

class MRCNNDetector(object):
    def __init__(self, model_path, num_classes, logs=None):

        class InferenceConfig2(Config):
            NAME = "object_detector"
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
            NUM_CLASSES = num_classes
            RPN_ANCHOR_SCALES = (32, 64, 128, 256, 384)

        config = InferenceConfig2()
        config.display()
            
        if logs is None:
            log_dir = os.path.join(os.environ['HOME'], '.ros/logs')
            logs = os.path.join(os.path.dirname(os.path.realpath(__file__)), log_dir)
            if not os.path.isdir(logs):
                os.mkdir(logs)
        
        self.__model = modellib.MaskRCNN(mode="inference", config=config, model_dir=logs)

        print("Loading pretrained model into memory")
        self.__model.load_weights(model_path, by_name = True)

        print("Successfully loaded pretrained model into memory")

    def detect(self, image, show_timer=False):

        if len(image.shape) < 3:
            print ('invalid image type')
            return None

        t_start = time.time()
        result = self.__model.detect([image], verbose = False)[0]
        t_pred = time.time() - t_start

        if show_timer:
            print ('Prediction time: {}'.format(t_pred))

        return result

    def plot_detection(self, image, result, labels):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure 
        
        figure = Figure()
        canvas = FigureCanvasAgg(figure)
        axes = figure.gca()

        colors = visualize.random_colors(len(labels))
        visualize.display_instances(image, result['rois'], result['masks'], result['class_ids'],
                                    labels, result['scores'], ax=axes, colors=colors)

        figure.tight_layout()
        canvas.draw()
        r = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

        x, y, w, h = figure.bbox.bounds
        r = r.reshape((int(h), int(w), 3))
        return r
