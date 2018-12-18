#!/usr/bin/env python

import os
import sys
import math
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
    def __init__(self, data_dir = None, class_labels = 'class.txt', \
                 labels = 'masks', background_folder = None):

        assert os.path.isdir(data_dir), 'Invalid dataset directory! {}'.format(data_dir)

        class_labels = os.path.join(data_dir, class_labels)
        assert os.path.isfile(class_labels), 'Class label textfile not found!{}'.format(class_labels)

        labels_dir = os.path.join(data_dir, labels)
        assert os.path.isdir(labels_dir), 'Dataset labels folder not found: {}'.format(labels_dir)

        image_dir = os.path.join(data_dir, 'images')
        assert os.path.isdir(image_dir), 'Dataset image folder not found: {}'.format(image_dir)
        
        #! read the object labels
        lines = self.read_textfile(class_labels)
        class_label_dict = {}
        for line in lines:
            name, label = line.split(' ')
            class_label_dict[int(label)] = name

        self.image_ids = []

        #! read all labels folder
        self.__dataset = []
        for index, dfile in enumerate(os.listdir(labels_dir)):
            fn, ext = os.path.splitext(dfile)
            if len(ext) is 0:
                im_path = os.path.join(image_dir, fn + '.jpg')
                la_path = os.path.join(labels_dir, fn)
                if os.path.isfile(im_path):
                    #! read all label images in the folder
                    label_lists = []
                    class_ids = []
                    for llist in os.listdir(la_path):
                        #! filename is class id
                        class_id, _ = os.path.splitext(llist)
                        label_lists.append(os.path.join(labels_dir, os.path.join(fn, llist)))
                        class_ids.append(int(class_id))
                        
                    self.__dataset.append({'image': im_path, 'labels': label_lists, 'class_id': class_ids})
                    self.image_ids.append(index)
                else:
                    print ('Image not found! {}'.format(im_path))
                    
        #! generate multiple instance
        self.__iou_thresh = 0.05
        self.__max_counter = 100
        self.__num_class = len(class_label_dict)
        self.__net_size = (800, 800)

        self.num_classes = len(class_label_dict) + 1
        self.source_class_ids = np.arange(0, len(lines)+1)

        self.image_info = {}
        for i in range(len(self.__dataset)):
            self.image_info[i] = {'source': np.arange(0, len(lines)+1)}

        self.__fetched_data = None
        self.__fetch_counter = 0

        self.DEBUG = False
        if self.DEBUG:
            self.__colors = np.random.randint(0, 255, (len(self.__dataset), 3))
            self.__fetch_counter = 0
            for i in range(len(self.__dataset)):
                print (self.__fetch_counter)
                self.__fetch_counter += 0
                self.load_image(i)
                self.load_mask(i)
        
    def get_dataset(self):
        return self.__dataset
        
    def load_image(self, image_id):
        self.__fetched_data = None
        self.__fetched_data = self.fetch_image_gt(index=image_id)
        return self.__fetched_data[0]

    def load_mask(self, image_id):
        masks = self.__fetched_data[1]
        class_id = self.__fetched_data[2]

        if self.DEBUG:
            img = self.__fetched_data[0].copy()
            for m in range(masks.shape[2]):
                # img = self.__fetched_data[0].copy()
                img[masks[:,:, m]] = self.__colors[m]
                
                if class_id[m] != 0:
                    print ([class_id[m], m])
            cv.imshow('mask', np.hstack([self.__fetched_data[0], img]))
            cv.waitKey(0)
                # cv.destroyAllWindows()
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
    
    def fetch_image_gt(self, index=None):
        
        #! TODO: read from set of images
        if self.__fetch_counter == self.__max_counter:
            print ('Loading Background')
            self.__fetch_counter = 0
            index = random.randint(0, len(self.__background)-1)
            im_bg = self.__background[index]['image']
            mk_bg = self.__background[index]['labels']
            class_ids = np.array([0])
            rects = np.zeros((1, 4, len(class_ids)), np.int0)
            rects[0, :, 0] = np.array([0, 0, im_bg.shape[1], im_bg.shape[0]], np.int0)
            return im_bg, mk_bg, class_ids, rects, None
        else:
            im_bg = None
            num_arguments = 1
            return self.argument(num_arguments, im_bg, index)
            
    def argument(self, num_proposals, im_bg, index=None):
        #! randomly select an index
        if index is None:
            index = random.randint(0, len(self.__dataset) - 1)
        im_path = self.__dataset[index]['image']
        label_paths = self.__dataset[index]['labels']
        class_ids = self.__dataset[index]['class_id']

        flip_flag = random.randint(-1, 2)
        
        #! read the rgb image
        im_rgb = cv.imread(im_path, cv.IMREAD_COLOR)
        if flip_flag != 2:
            im_rgb = cv.flip(im_rgb, flip_flag)

        masks = np.zeros((im_rgb.shape[0], im_rgb.shape[1], self.__num_class), np.bool)
        rects = np.zeros((1, 4, self.__num_class), np.int0)
        ordered_class_ids = np.zeros((self.__num_class), np.int0)

        for cls_id, lpath in zip(class_ids, label_paths):
            mk = cv.imread(lpath, cv.IMREAD_ANYDEPTH)
            
            if flip_flag != 2:
                mk = cv.flip(mk, flip_flag)

            _, contour = self.edge_contour_points(mk.copy())

            rect = cv.boundingRect(contour[0])
            rects[0, :, cls_id-1] = np.array(rect)

            mk = mk.astype(np.bool)
            masks[:, :, cls_id-1] = mk            
            ordered_class_ids[cls_id-1] = cls_id

        # overall bounding rect
        minx = miny = 1E9
        maxx = maxy = 0
        # for rect in rects:
        for i, cls_id in enumerate(ordered_class_ids):
            if cls_id == 0:
                continue
            x,y,w,h = rects[0, :, i]
            y2 = y+h
            x2 = x+w
            minx = x if x < minx else minx
            miny = y if y < miny else miny
            maxx = x2 if x2 > maxx else maxx
            maxy = y2 if y2 > maxy else maxy


        padding = random.randint(0, 20)
        diffx = im_rgb.shape[1]-(maxx-minx) if im_rgb.shape[1]-(maxx-minx) > padding else padding
        diffy = im_rgb.shape[0]-(maxy-miny) if im_rgb.shape[0]-(maxy-miny) > padding else padding
        padx = random.randint(padding, diffx)
        pady = random.randint(padding, diffy)
        
        minx = minx-padx if minx-padx > 0 else 0
        miny = miny-pady if miny-pady > 0 else 0
        maxx = maxx+padx if maxx+padx < im_rgb.shape[1]-1 else im_rgb.shape[1]-1
        maxy = maxy+pady if maxy+pady < im_rgb.shape[0]-1 else im_rgb.shape[0]-1

        #! crop all to new size
        if random.randint(0, 1):
            try:
                im_rgb = im_rgb[miny:maxy, minx:maxx]
                masks = masks[miny:maxy, minx:maxx]
            except:
                pass

        return im_rgb, masks, ordered_class_ids, rects, None

    def crop_image_dimension(self, image, rect, width, height):
        x = int((rect[0] + rect[2]/2) - width/2)
        y = int((rect[1] + rect[3]/2) - height/2)
        w = width
        h = height

        ## center 
        cx, cy = (rect[0] + rect[2]/2.0, rect[1] + rect[3]/2.0)
        shift_x, shift_y = (random.randint(0, int(w/2)), random.randint(0, int(h/2)))
        cx = (cx + shift_x) if random.randint(0, 1) else (cx - shift_x)
        cy = (cy + shift_y) if random.randint(0, 1) else (cy - shift_y)
        
        nx = int(cx - (w / 2))
        ny = int(cy - (h / 2))
        nw = int(w)
        nh = int(h)

        if nx > x:
            nx = x
            nw -=  np.abs(nx - x)
        if ny > y:
            ny = y
            nh -=  np.abs(ny - y)
        if nx + nw < x + w:
            nx += ((x+w) - (nx+nw))
        if ny + nh < y + h:
            ny += ((y+h) - (ny+nh))

        x = nx; y = ny; w = nw; h = nh

        x = 0 if x < 0 else x
        y = 0 if y < 0 else y
        w = ((w - (w + x) - image.shape[1])) if x > image.shape[1] else w
        h = ((h - (h + y) - image.shape[0])) if y > image.shape[0] else h

        roi = image[int(y):int(y+h), int(x):int(x+w)].copy()
        new_rect = [int(rect[0] - x), int(rect[1] - y), rect[2], rect[3]]

        return roi, new_rect

    def resize_image_and_labels(self, image, rects, resize):
        img_list = []
        resize_rects = []
        for rect in rects:
            img = cv.resize(image, resize, cv.INTER_CUBIC)
            img_list.append(img)
            # resize label
            ratio_x = np.float32(image.shape[1]) / np.float32(img.shape[1])
            ratio_y = np.float32(image.shape[0]) / np.float32(img.shape[0])
            
            x = np.float32(rect[0])
            y = np.float32(rect[1])
            w = np.float32(rect[2])
            h = np.float32(rect[3])
            
            xt = x / ratio_x
            yt = y / ratio_y
            xb = (x + w) / ratio_x
            yb = (y + h) / ratio_y
                
            rect_resize = (int(xt), int(yt), int(xb - xt), int(yb - yt))
            resize_rects.append(rect_resize)
        return img, resize_rects


    def edge_contour_points(self, im_mask):
        if im_mask is None:
            return im_mask, None

        im_mask2 = im_mask.copy()
        if len(im_mask2.shape) == 3:
            im_mask2 = cv.cvtColor(im_mask, cv.COLOR_BGR2GRAY)

        _, im_mask2 = cv.threshold(im_mask2, 127, 255, 0)        
        _ , contours, _ = cv.findContours(im_mask2, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)
        
        #! remove noisy contours
        min_area = 16**2
        max_area = 1E9
        contour_obj = None
        for contour in contours:
            area = cv.contourArea(contour)
            if area < min_area:
                continue
            max_area, contour_obj = (area, contour) if area < max_area \
                                    else (max_area, contour_obj)
        
        if max_area < min_area:
            return im_mask, None
        else:
            return im_mask, [contour_obj]

            
    @classmethod
    def read_textfile(cls, filename):
        lines = [line.rstrip('\n')
                 for line in open(filename)                                                                             
        ]
        return np.array(lines)

    @classmethod
    def read_images(cls, **kwargs):
        im_rgb = cv.imread(kwargs['image'], cv.IMREAD_COLOR)
        im_dep = cv.imread(kwargs['depth'], cv.IMREAD_ANYCOLOR)
        im_mask = cv.imread(kwargs['mask'], cv.IMREAD_GRAYSCALE)
        return [im_rgb, im_dep, im_mask]
        
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

    def get_images_in_folder(self, directory):
        files = os.listdir(directory)
        images = [f for f in files if os.path.splitext(f)[1] == '.jpg' or os.path.splitext(f)[1] == '.png']
        return images

    
def main(argv):
    if len(argv) < 2:
        raise ValueError('Provide image list.txt')
    smi = Dataloader(argv[1])

if __name__ == '__main__':
    main(sys.argv)
