#!/usr/bin/env python

import os
import sys
import random
import json
import cv2 as cv
import numpy as np
from tqdm import tqdm

"""
Class for computing intersection over union(IOU)
"""
class JaccardCoeff:

    def iou(self, a, b):
        i = self.__intersection(a, b)
        if i == 0:
            return 0
        aub = self.__area(self.__union(a, b))
        anb = self.__area(i)
        area_ratio = self.__area(a)/self.__area(b)        
        score = anb/aub
        score /= area_ratio
        return score
        
    def __intersection(self, a, b):
        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0]+a[2], b[0]+b[2]) - x
        h = min(a[1]+a[3], b[1]+b[3]) - y
        if w < 0 or h < 0:
            return 0
        else:
            return (x, y, w, h)
        
    def __union(self, a, b):
        x = min(a[0], b[0])
        y = min(a[1], b[1])
        w = max(a[0]+a[2], b[0]+b[2]) - x
        h = max(a[1]+a[3], b[1]+b[3]) - y
        return (x, y, w, h)

    def __area(self, rect):
        return np.float32(rect[2] * rect[3])

class Dataloader(object):
    def __init__(self, data_dir = None, objects_list = 'objects.txt', \
                 filename = 'train.txt', background_folder = None):
        if data_dir is None:
            raise ValueError('Provide dataset directory')
        if filename is None:
            raise ValueError('Provide image train.txt')
        if objects_list is None or not os.path.isfile(os.path.join(data_dir, objects_list)):
            raise ValueError('Provide image objects.txt')

        self.__dataset_dir = data_dir
        self.__objects = None
        self.__dataset = {}

        self.read_data_from_textfile(objects_list, filename)

        #! generate multiple instance
        self.__iou_thresh = 0.05
        self.__max_counter = 100
        self.__num_class = len(self.__dataset)

        print ("Checking all the data")
        is_check = False
        status = self.sneek_dataset() if is_check else True
        if status:
            print ("Successfully checked the data")
        else:
            print ("Seems data are missing. Please check")
            sys.exit()
        
        self.image_ids = [0, 1, 3, 4, 5] ##! fix this
        self.num_classes = len(self.__dataset) + 1
        self.source_class_ids = {'': [0], 'handheld_objects': [0, 1, 2, 3, 4, 5, 6]}
        
        self.__fetched_data = None

        ##! prepare background images

        ##! hardcoded
        background_folder = '/home/krishneel/Desktop/background/'
        if not background_folder is None:
            print ("background %s" % background_folder)
            self.__image_lists = self.read_files_in_folder(background_folder)
            

    def get_dataset(self):
        return self.__dataset
        
    def load_image(self, image_id):
        self.__fetched_data = None
        self.__fetched_data = self.fetch_image_gt()
        return self.__fetched_data[0]

    def load_mask(self, image_id):
        masks = self.__fetched_data[1].transpose((1, 2, 0))
        class_id = self.__fetched_data[2]

        debug = False
        if debug:
            for m in range(masks.shape[2]):
                img = self.__fetched_data[0].copy()
                img[masks[:,:, m]] = 255

                print (class_id[m])
                cv.imshow('mask', img)
                cv.waitKey(0)
            cv.destroyAllWindows()

        return masks, class_id
        
        
    def image_info(self, image_id):
        ret_val = {
            'source': 'handheld_objects',
            'width': self.__fetched_data[0].shape[1],
            'height': self.__fetched_data[0].shape[0],
            'bbox': self.__fetched_data[3],
            'path': self.__fetched_data[4]
        }
        return ret_val
    
    def fetch_image_gt(self):
        
        #! TODO: read from set of images
        # im_bg = cv.imread("/home/krishneel/Desktop/image.jpg")
        
        im_bg = self.generate_background_image(self.__image_lists)
        
        num_arguments = random.randint(1, 10)
        return self.argument(num_arguments, im_bg, ret_binary = True)

        
    @classmethod
    def edge_contour_points(self, im_mask):
        if im_mask is None:
            return im_mask, None

        #! smooth the edges in mask
        im_mask = cv.GaussianBlur(im_mask, (21, 21), 11.0)
        im_mask[im_mask > 150] = 255

        im_mask2 = im_mask.copy()
        if len(im_mask2.shape) == 3:
            im_mask2 = cv.cvtColor(im_mask, cv.COLOR_BGR2GRAY)

        _, im_mask2 = cv.threshold(im_mask2, 127, 255, 0)
        _ , contours, _ = cv.findContours(im_mask2, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)

        #! remove noisy contours
        max_area = 0
        contour_obj = None
        
        for contour in contours:
            area = cv.contourArea(contour)
            max_area, contour_obj = (area, contour) if area > max_area \
                                    else (max_area, contour_obj)

        if max_area < 16 ** 2:
            return im_mask, None
        else:
            return im_mask, [contour_obj]

            
    @classmethod
    def read_textfile(self, filename):
        lines = [line.rstrip('\n')
                 for line in open(filename)                                                                             
        ]
        return np.array(lines)

    @classmethod
    def read_images(self, **kwargs):
        im_rgb = cv.imread(kwargs['image'], cv.IMREAD_COLOR)
        im_dep = cv.imread(kwargs['depth'], cv.IMREAD_ANYCOLOR)
        im_mask = cv.imread(kwargs['mask'], cv.IMREAD_COLOR)
        return [im_rgb, im_dep, im_mask]
    
    def read_data_from_textfile(self, objects_list, filename):
        self.__objects = self.read_textfile(os.path.join(self.__dataset_dir, objects_list))

        print ('Objects: {}'.format(self.__objects))
        
        for obj_label in self.__objects:
            obj = obj_label.split(' ')[0]
            label = int(obj_label.split(' ')[1])
            
            fn = os.path.join(self.__dataset_dir, os.path.join(obj, filename))
            if not os.path.isfile(fn):
                raise Exception('Missing data train.txt')
            lines = self.read_textfile(fn)

            datas = []
            for line in lines:
                pline = line
                if not os.path.isfile(pline):
                    pline = os.path.join(self.__dataset_dir, line)
                with open(pline) as f:
                    anno_data = json.load(f)

                #! check if file exits
                if not os.path.isfile(os.path.join(self.__dataset_dir, anno_data['image'])):
                    print ("File not found at {}".format(os.path.join(self.__dataset_dir, anno_data['image'])))
                    sys.exit()
                
                datas.append({
                    'image': os.path.join(self.__dataset_dir, anno_data['image']),
                    'depth' : os.path.join(self.__dataset_dir, anno_data['depth']),
                    'mask' : os.path.join(self.__dataset_dir, anno_data['mask'])
                })
            
            self.__dataset[str(obj)] = {'data_list': np.array(datas), 'label': label}
            
    def argument(self, num_proposals, im_bg, im_mk = None, mrect = None, ret_binary = False):
        im_y, im_x, _ = im_bg.shape
        flag_position = []
        img_output = im_bg.copy()

        mask_output = np.zeros((im_y, im_x, 1), np.uint8)
        if not im_mk is None:
            mask_output = im_mk.copy()
        if not mrect is None:
            flag_position.append(mrect)

        labels = []
        im_path = None

        if ret_binary:
            binary_masks = []
        for index in range(num_proposals):
            while(True):
                rindex = random.randint(0, len(self.__dataset) - 1)
                object_name = list(self.__dataset.keys())[rindex]
                label = self.__dataset[object_name]['label']
                
                im_index = random.randint(0, len(self.__dataset[object_name]['data_list']) - 1)
                im_list = self.__dataset[object_name]['data_list'][im_index]
                [image, depth, mask] = self.read_images(**im_list)

                im_path = im_list['image']
                
                mask, contours = self.edge_contour_points(mask)
                if not contours is None:
                    rect = np.array(cv.boundingRect(contours[0]))
                    x,y,w,h = rect
                    # cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 255), 1)            
                    break
            
            flip_flag = random.randint(-1, 2)
            if flip_flag > -2 and flip_flag < 2:
                rect_copy = rect.copy()
                image, rect = self.flip_image(image, [rect], flip_flag)
                mask, _ = self.flip_image(mask, [rect_copy], flip_flag)
                x,y,w,h = rect[0]
            
            im_roi = image[y:y+h, x:x+w].copy()
            im_msk = mask[y:y+h, x:x+w].copy()
            
            resize_flag = random.randint(0, 1)
            if resize_flag:
                scale = random.uniform(1.0, 2.2)
                w = int(w * scale)
                h = int(h * scale)
                im_roi = cv.resize(im_roi, (int(w), int(h)))
                im_msk = cv.resize(im_msk, (int(w), int(h)))
                rect  = np.array([x, y, w, h], dtype=np.int)
            
            cx, cy = random.randint(0, im_x - 1), random.randint(0, im_y-1)
            cx = cx - ((cx + w) - im_x) if cx + w > im_x - 1 else cx
            cy = cy - ((cy + h) - im_y) if cy + h > im_y - 1 else cy
            nrect = np.array([cx, cy, w, h])

            ##! and-ing to remove bg pixels
            im_msk2 = im_msk.copy()
            im_msk[im_msk < 127] = 0
            im_roi = cv.bitwise_and(im_roi, im_msk)
            
            counter = 0
            position_found = True
            if len(flag_position) > 0:
                jc = JaccardCoeff()
                for bbox in flag_position:
                    if jc.iou(bbox, nrect) > self.__iou_thresh and position_found:
                        is_ok = True
                        while True:
                            cx, cy = random.randint(0, im_x - 1), random.randint(0, im_y-1)
                            cx = cx - ((cx + w) - im_x) if cx + w > im_x - 1 else cx
                            cy = cy - ((cy + h) - im_y) if cy + h > im_y - 1 else cy
                            nrect = np.array([cx, cy, w, h])
                            for bbox2 in flag_position:
                                if jc.iou(bbox2, nrect) > self.__iou_thresh:
                                    is_ok = False
                                    break
                            if is_ok:
                                break

                            counter += 1
                            if counter > self.__max_counter:
                                position_found = False
                                break
            if position_found:
                im_roi = cv.bitwise_and(im_roi, im_msk)

                if ret_binary:
                    bmask = np.zeros((mask_output.shape[0], mask_output.shape[1]), np.bool)
                for j in range(h):
                    for i in range(w):
                        nx, ny = i + cx, j + cy
                        if im_msk[j, i, 0] > 0 and nx < im_x and ny < im_y:
                            img_output[ny, nx] = im_roi[j, i]
                            mask_output[ny, nx] = label ##! check this
                            
                            if ret_binary:
                                bmask[ny, nx] = True
                if ret_binary:
                    binary_masks.append(bmask)
                flag_position.append(nrect)
                labels.append(label)

        ###! debug
        debug = False
        if debug:
            print ("label {}".format(labels))
            z = np.hstack((im_msk, im_roi))
            cv.imshow('imask', z)
            
            for r in flag_position:
                x,y,w,h = r
                cv.rectangle(img_output, (x,y), (x+w, h+y), (0, 255, 0), 3)
                cv.namedWindow('roi', cv.WINDOW_NORMAL)
                cv.imshow('roi', img_output)

            im_flt = mask_output.astype(np.float32)
            # im_flt = cv.normalize(im_flt, 0, 1, cv.NORM_MINMAX)
            im_flt /= len(self.__dataset)
            im_flt *= 255.0
            im_flt = im_flt.astype(np.uint8)
            im_flt = cv.applyColorMap(im_flt, cv.COLORMAP_JET)
            
            cv.imshow('mask2', im_flt)
            mask_output2 = mask_output.copy() * 255
            # cv.imshow('mask', mask_output2)
            cv.waitKey(0)

        ###! end-debug

        if ret_binary:
            return (img_output, np.array(binary_masks), np.array(labels), np.array(flag_position), im_path)
        else:
            return (img_output, mask_output, np.array(labels), np.array(flag_position), im_path)

    """
    Function flip image and rect around given axis
    """     
    def flip_image(self, image, rects, flip_flag = -1):
        im_flip = cv.flip(image, flip_flag)
        flip_rects = []
        for rect in rects:
            pt1 = (rect[0], rect[1])
            pt2 = (rect[0] + rect[2], rect[1] + rect[3])
            if flip_flag is -1:
                pt1 = (image.shape[1] - pt1[0] - 1, image.shape[0] - pt1[1] - 1)
                pt2 = (image.shape[1] - pt2[0] - 1, image.shape[0] - pt2[1] - 1)
            elif flip_flag is 0:
                pt1 = (pt1[0], image.shape[0] - pt1[1] - 1)
                pt2 = (pt2[0], image.shape[0] - pt2[1] - 1)
            elif flip_flag is 1:
                pt1 = (image.shape[1] - pt1[0] - 1, pt1[1])
                pt2 = (image.shape[1] - pt2[0] - 1, pt2[1])

            x = min(pt1[0], pt2[0])
            y = min(pt1[1], pt2[1])
            w = np.abs(pt2[0] - pt1[0])
            h = np.abs(pt2[1] - pt1[1])

            x = 0 if x < 0 else x
            y = 0 if y < 0 else y

            flip_rect = [x, y, w, h]
            flip_rects.append(flip_rect)
        return im_flip, flip_rects

    def sneek_dataset(self):
        cv.namedWindow('image', cv.WINDOW_AUTOSIZE)
        print ("Sneeking through the dataset")
        for obj in self.__dataset:
            print ("\tchecking: %s" % obj)
            obj_data_list = self.__dataset[obj]['data_list']
            for im_list in obj_data_list:
                [im_rgb, im_dep, im_mask] = self.read_images(**im_list)

                if im_rgb is None or im_mask is None:
                    print ("Image Not found")
                    print (im_list)
                    sys.exit()
                
                im_mask2 = im_mask.copy()
                im_mask, contours = self.edge_contour_points(im_mask)

                # file the gaps
                if not contours is None:
                    cv.drawContours(im_mask, contours, -1, (255, 255, 255), -1)

                rgb_mask = cv.bitwise_and(im_rgb, im_mask)
                if not contours is None:
                    cv.drawContours(rgb_mask, contours, -1, (0, 255, 0), 3)

                    # bounding rectangle
                    x, y, w, h = cv.boundingRect(contours[0])
                    cv.rectangle(rgb_mask, (x, y), (x+w, y+h), (255, 0, 255), 1)

                    #! save the json annotation file
                    
                # debug
                alpha = 0.3
                cv.addWeighted(im_rgb, alpha, rgb_mask, 1.0 - alpha, 0, rgb_mask)
                
                rgb_mask2 = cv.bitwise_and(im_rgb, im_mask2)
                z = np.hstack((rgb_mask, rgb_mask2))
                cv.imshow('image', z)
                cv.waitKey(20)
        cv.destroyAllWindows()
        return True


    def check_dataset(self):

        for obj in self.__dataset:
            print ("\tchecking: %s" % obj)
            obj_data_list = self.__dataset[obj]['data_list']
            for im_list in obj_data_list:
                [im_rgb, im_dep, im_mask] = self.read_images(**im_list)
                if im_rgb is None or im_mask is None:
                    print (im_list)
                    sys.exit()

    def generate_test_image(self, num_imgs):
        save_dir = os.path.join(os.environ['HOME'], 'Desktop/test_images')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in range(num_imgs):
            gt_data = self.fetch_image_gt()

            img = gt_data[0]
            im_name = str(time.time()) + '.jpg'
            cv.imwrite(os.path.join(save_dir, im_name), img)


    def read_files_in_folder(self, path_to_folder, extension = '.jpg'):
        image_lists = [
            os.path.join(path_to_folder, ifile) 
            for ifile in os.listdir(path_to_folder)
            if ifile.endswith(extension)
        ]
        print ('background images: {} {}'.format(image_lists, len(image_lists)))
        return image_lists
        
    def generate_background_image(self, image_lists):
        index = random.randint(0, len(image_lists) - 1)
        im_bg = cv.imread(image_lists[index], cv.IMREAD_COLOR)

        #! min crop size
        min_w = im_bg.shape[1] / 2
        min_h = im_bg.shape[0] / 2

        rwidth = random.randint(min_w, im_bg.shape[1])
        rheight = random.randint(min_h, im_bg.shape[0])
        rx = random.randint(0, min_w)
        ry = random.randint(0, min_h)

        rx -= (rx + rwidth - im_bg.shape[1]) if rx + rwidth > im_bg.shape[1] else 0
        ry -= (ry + rheight - im_bg.shape[0]) if ry + rheight > im_bg.shape[0] else 0
        
        #! crop image
        size = im_bg.shape
        im_bg = im_bg[ry:ry + rheight, rx:rx + rwidth].copy()        
        im_bg = cv.resize(im_bg, (size[1], size[0]))

        debug = False
        if debug:
            cv.imshow("back", im_bg)
            cv.waitKey(0)
            cv.destroyAllWindows()
            
        return im_bg
    
def main(argv):
    if len(argv) < 2:
        raise ValueError('Provide image list.txt')

    smi = Dataloader(argv[1], argv[2], argv[3], argv[4])
    

if __name__ == '__main__':
    main(sys.argv)
