#!/usr/bin/env python

###########################################################################
## Copyright (C) 2018 by Krishneel Chaudhary @ JSK Lab,
## The University of Tokyo, Japan
###########################################################################

import os
import sys
import random
import numpy as np
import cv2 as cv
import rospy
import time
import json

from detector import MRCNNDetector

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point32, PolygonStamped
from mask_rcnn_ros.msg import RectArray
from mask_rcnn_ros.srv import *

class ObjectDetector(MRCNNDetector):

    def __init__(self):
        
        self.__model = rospy.get_param('~model', None)
        self.__class_labels = rospy.get_param('~class_labels', None)
        self.__prob_thresh = rospy.get_param('~detection_threshold', 0.5)
        self.__is_service = rospy.get_param('~is_service', False)
        self.__debug = rospy.get_param('~debug', True)

        assert os.path.isfile(self.__model), 'Trained model file not found! {}'.format(self.__model)
        assert os.path.isfile(self.__class_labels), 'Class labels file not found! {}'.format(self.__class_labels)
        
        #! initialize the detector
        self.__detector_init = True

        #! read the object class name and labels
        lines = [line.rstrip('\n') for line in open(self.__class_labels)]

        self.__objects_meta = {}
        for line in lines:
            object_name, object_label = line.split()
            object_label = int(object_label)
            #! color is for visualization
            color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            self.__objects_meta[object_label] = [object_name, color]

        self.pub_detection = rospy.Publisher('/object_detector/rects', RectArray, queue_size=1)

        if not self.__is_service:
            rospy.loginfo('RUNNING MASK-RCNN DETECTOR AS PUBLISHER NODE')
            self.subscribe()
        else:
            rospy.loginfo('RUNNING MASK-RCNN DETECTOR AS SERVICE')
            self.service()
            
        
    def run_detector(self, image, header=None):

        result = self.detect(image)

        rois = result['rois']
        masks = result['masks']
        class_ids = result['class_ids']
        scores = result['scores']
        
        rect_array = RectArray()
        for m in range(masks.shape[2]):
            if scores[m] > self.__prob_thresh:
                object_name, color = self.__objects_meta[class_ids[m]]

                y1, x1, y2, x2 = rois[m]

                poly = PolygonStamped()
                poly.polygon.points.append(Point32(x=x1, y=y1))
                poly.polygon.points.append(Point32(x=x2, y=y2))
                if header is not None:
                    poly.header = header
                
                rect_array.polygon.append(poly)
                rect_array.labels.append(int(class_ids[m]))
                rect_array.likelihood.append(scores[m])
                rect_array.names.append(object_name)
                # rect_array.image.append(masks[:,:,m])

                im_mask = np.zeros(image.shape[:2], np.int64)
                im_mask[masks[:,:,m]==True] = np.int64(class_ids[m])

                ## temp
                for j in range(im_mask.shape[0]):
                    for i in range(im_mask.shape[1]):
                        rect_array.indices.append(im_mask[j, i])

                if self.__debug:
                    cv.putText(image, object_name, (x1, y1),
                               cv.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2, cv.LINE_8)
                    cv.rectangle(image, (x1, y1), (x2, y2), color, 3)
                    image[masks[:,:,m]] = color

        """
        for j in range(im_mask.shape[0]):
            for i in range(im_mask.shape[1]):
                rect_array.indices.append(im_mask[j, i])
        """
            
        if self.__debug:
            print ('Scores: {} {} {}'.format(scores, class_ids, image.shape))
            wname = 'image'
            cv.namedWindow(wname, cv.WINDOW_NORMAL)
            cv.imshow(wname, image)
            if cv.waitKey(3) == 27:
                cv.destroyAllWindows()
                sys.exit()

        return rect_array
            
    def convert_to_cv_image(self, image_msg):

        if image_msg is None:
            return None
        
        width = image_msg.width
        height = image_msg.height
        channels = int(len(image_msg.data) / (width * height))

        encoding = None
        if image_msg.encoding.lower() in ['rgb8', 'bgr8']:
            encoding = np.uint8
        elif image_msg.encoding.lower() == 'mono8':
            encoding = np.uint8
        elif image_msg.encoding.lower() == '32fc1':
            encoding = np.float32
            channels = 1

        cv_img = np.ndarray(shape=(image_msg.height, image_msg.width, channels),
                            dtype=encoding, buffer=image_msg.data)

        if image_msg.encoding.lower() == 'mono8':
            cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2GRAY)
        else:
            cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2BGR)

        return cv_img


    def callback(self, image_msg):
        
        if self.__detector_init:
            rospy.loginfo('Loading pretrained model into memory')
            MRCNNDetector.__init__(self, model_path=self.__model, logs=None, \
                                   num_classes=len(self.__objects_meta) + 1)
            self.__detector_init = False
            rospy.loginfo('Detector is successfully initialized')
            
        image = self.convert_to_cv_image(image_msg)
        if not image is None:
            
            rect_array = self.run_detector(image, image_msg.header)
            rect_array.header = image_msg.header

            if not self.__is_service:
                self.pub_detection.publish(rect_array)
            else:
                return MaskRCNNResponse(rect_array)
        else:
            rospy.logwarn('Empty image')
        
    def subscribe(self):
        rospy.Subscriber('image', Image, self.callback, queue_size = 1)

    def service_handler(self, request):
        return self.callback(request.image)
        
    def service(self):
        rospy.loginfo('SETTING UP SRV')
        srv = rospy.Service('mask_rcnn_object_detector', MaskRCNN, self.service_handler)

def main(argv):
    try:
        rospy.init_node('object_detector', anonymous = False)
        detector = ObjectDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logfatal('error with object_detector setup')
        sys.exit()

if __name__ == '__main__':
    main(sys.argv)
