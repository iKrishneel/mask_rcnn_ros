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

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from train.custom_dataloader import Dataloader
from detector import MRCNNDetector

class BootstrapDetector(MRCNNDetector, Dataloader):
    def __init__(self):
        self.__bridge = CvBridge()

        self.__pretrained_model = rospy.get_param('~pretrained_model', None)
        if self.__pretrained_model is None:
            rospy.logfatal('The absolute path to the package needs to be set')
            # sys.exit()

        direct = '/home/krishneel/Documents/program/object_detector/scripts/logs/handheld_objects20180910T1400/'
        model_name = 'mask_rcnn_handheld_objects_0094.h5'
        self.__pretrained_model = direct + model_name

        #! read the ground truth dataset
        self.__dataset_dir = '/home/krishneel/Documents/datasets/objects610/'
        Dataloader.__init__(self, data_dir = self.__dataset_dir)
        self.__dataset = self.get_dataset()

        if self.__dataset is None:
            print ('Dataset not found')
            sys.exit()
            
        #! initialize the detector
        MRCNNDetector.__init__(self, self.__pretrained_model, direct)
        
        ###! test only
        self.__objects = ['',
                          '001', '002', '003', '004', '005', '006', '007', '008', '009', '010',
                          '011', '012', '013', '014', '015', '016', '017', '018', '019', '020',
                          '021', '022', '023', '024', '025'
        ]
        self.__objects = ['', 'cup-star', 'cup', 'shampoo', 'curry', 'macha', 'latte', 'banana']
        
        self.__colors = []
        for i in range(len(self.__objects)):
            self.__colors.append([
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            ])

            
        cv.namedWindow('image', cv.WINDOW_NORMAL)
        self.bootstrap_detector()
        sys.exit()


    def create_folder(self, directory):
        if not os.path.isdir(directory):
            os.mkdir(directory)
    
    def bootstrap_detector(self):

        self.__dataset_dir += os.sep if self.__dataset_dir[-1] != os.sep else ""
        
        save_dir = os.path.join(self.__dataset_dir, 'mask_bootstrap')
        self.create_folder(save_dir)
            
        folder_name = 'mask_' + str(int(time.time()))
        save_dir = os.path.join(save_dir, folder_name)
        self.create_folder(save_dir)
        
        for key in self.__dataset.keys():

            write_dir = os.path.join(save_dir, key)
            self.create_folder(write_dir)
                
            write_jdir = os.path.join(write_dir, 'annotations')
            write_mdir = os.path.join(write_dir, 'mask')
            self.create_folder(write_jdir)
            self.create_folder(write_mdir)

            text_file = open(os.path.join(write_dir, 'train.txt'), 'w')
            
            object_dataset = self.__dataset[key]['data_list']
            for img_list in object_dataset:

                print (img_list)
                
                im_rgb, _, im_mask = self.read_images(**img_list)

                img = im_rgb.copy()
                if len(im_mask.shape) == 3:
                    im_mask = cv.cvtColor(im_mask, cv.COLOR_BGR2GRAY)
                
                im_rgb2, pred_mask = self.run_detector(im_rgb, label = self.__dataset[key]['label'])
                im_rgb[im_mask > 0] = 127

                new_mask = self.pixelwise_iou(im_mask, pred_mask.copy())
                new_mask = pred_mask
                im_rgb2[new_mask > 0] = self.__colors[self.__dataset[key]['label']]

                #! write the mask
                filename = img_list['mask'].split(os.sep)[-1]
                mk_p = os.path.join(write_mdir, filename)
                cv.imwrite(mk_p, new_mask)

                #! write annotation
                anno = {
                    'image': img_list['image'].replace(self.__dataset_dir, ''),
                    'depth': img_list['depth'].replace(self.__dataset_dir, ''),
                    'mask': mk_p.replace(self.__dataset_dir, ''),
                    'label': self.__dataset[key]['label'],
                    'bbox': [0, 0, 0, 0]
                }
                
                json_fn = os.path.join(write_jdir,  os.path.splitext(filename)[0] + '.json')
                with open(json_fn, 'w') as f:
                    json.dump(anno, f)

                text_file.write(json_fn.replace(self.__dataset_dir, '') + "\n")

                im_rgb2 = cv.addWeighted(im_rgb2, 0.5, img, 0.5, 0)
                
                z = np.hstack((im_rgb, im_rgb2))

                cv.imshow('image', z)
                if cv.waitKey(3) == 27:
                    cv.destroyAllWindows()
                    # sys.exit()
                    break
                    
            text_file.close()
            
    def pixelwise_iou(self, gt_mask, pred_mask):

        gt_mask[gt_mask > 150] = 255
        gt_mask[gt_mask <= 150] = 0
        pred_mask[pred_mask > 0] = 255

        in_mask = (gt_mask & pred_mask)
        un_mask = (gt_mask | pred_mask)

        _, in_counts = np.unique(in_mask, return_counts = True)
        _, un_counts = np.unique(un_mask, return_counts = True)

        iou_score = 0
        if len(in_counts) > 1 and len(un_counts) > 1:
            in_count = in_counts[1]
            un_count = un_counts[1]
                
            iou_score = float(in_count) / float(un_count)
            
        print ('\tiou: {}\n'.format(iou_score))
        # mask = np.hstack((in_mask, un_mask))
        # cv.imshow('mask', mask)
        new_mask = gt_mask
        if iou_score >= 0.75:
            new_mask = pred_mask
        elif iou_score > 0.4 and iou_score < 0.75:
            new_mask = un_mask
            
        return new_mask

    
    def run_detector(self, image, label = None):
        result = self.detect(image)

        rois = result['rois']
        masks = result['masks']
        class_ids = result['class_ids']
        scores = result['scores']
        
        # print ('Scores: {} {} {}'.format(scores, class_ids, label))
        
        img = image.copy()
        pred_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        for m in range(masks.shape[2]):
            if scores[m] > 0.80 and (True if label is None else class_ids[m] == label):
                y1, x1, y2, x2 = rois[m]
                cv.putText(img, self.__objects[class_ids[m]], (x1, y1), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2, cv.LINE_8)
                # cv.rectangle(img, (x1, y1), (x2, y2), self.__colors[class_ids[m]], 3)
                cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # img[masks[:,:,m]] = self.__colors[class_ids[m]]
                pred_mask[masks[:,:,m]] = 255
                
        return img, pred_mask        

    def convert_to_cv_image(self, image_msg):
        try:
            return self.__bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except Exception as e:
            rospy.logerr('%s' %e)
        return None

    
    def callback(self, image_msg):
        self.run_detector(image_msg)
    
    def subscribe(self):
        rospy.Subscriber('image', Image, self.callback, tcp_nodelay=True)


def main(argv):
    try:
        rospy.init_node('object_detector', anonymous = False)
        detector = BootstrapDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logfatal('error with object_detector setup')
        sys.exit()

if __name__ == '__main__':
    main(sys.argv)
