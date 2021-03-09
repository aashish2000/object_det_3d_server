"""
Images must be in ./Kitti/testing/image_2/ and camera matricies in ./Kitti/testing/calib/

Uses YOLO to obtain 2D box, PyTorch to get 3D box, plots both

SPACE bar for next image, any other key to exit
"""


from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import Model, ClassAverages
from yolo.yolo import cv_Yolo

import os
import time

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg

def object_info(corners):
    width_arr = []
    height_arr = []
    length_arr = []

    for pt in corners:
        width_arr.append(pt[0])
        height_arr.append(pt[1])
        length_arr.append(abs(pt[2]))

    width_arr.sort()
    height_arr.sort()
    length_arr.sort()

    print(width_arr)
    print(height_arr)
    print(length_arr)

    width = 0
    height = 0
    length = 0
    # width = abs(max(width_arr) - min(width_arr))
    # height = abs(max(height_arr) - min(height_arr))
    # length = abs(max(length_arr) - min(length_arr))

    for i in range(4):
        width += abs(width_arr[4+i] - width_arr[i])
        height += abs(height_arr[4+i] - height_arr[i])
        length += abs(length_arr[4+i] - length_arr[i])
    
    width /= 4
    length /= 4
    height /= 4.2

    camera_dist = abs(abs(length_arr[-1]) - length) - 0.2
    print(camera_dist)

    print("Width (metres):",width)
    print("Height (metres):", height)
    print("Length (metres):", length)
    print("Distance from Camera (metres):", camera_dist)

    return([width,height,length,camera_dist])

def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)
    corner1_2d = box_2d[0]
    corner2_2d = box_2d[1]

    pt1 = corner1_2d
    pt2 = (corner1_2d[0], corner2_2d[1])
    pt3 = corner2_2d
    pt4 = (corner2_2d[0], corner1_2d[1])
    # print("Pred 3D Corners: ",X)

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    corners = plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location, corners

def initialize_model():
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        exit()
    else:
        my_vgg = vgg.vgg19_bn(pretrained=True)
        # TODO: load bins from file or something
        model = Model.Model(features=my_vgg.features, bins=2)
        checkpoint = torch.load(weights_path + '/%s'%model_lst[-1],map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # load yolo
    yolo_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    yolo = cv_Yolo(yolo_path)

    averages = ClassAverages.ClassAverages()

    # TODO: clean up how this is done. flag?
    angle_bins = generate_bins(2)

    return(yolo, model, averages, angle_bins)


def object_detection_3d(img_id, yolo, model, averages, angle_bins):

    start_time = time.time()

    img_file = img_id

    # P for each frame
    calib_file = img_id.rsplit(".",1)[0] + ".txt"
    print(img_file,calib_file)
    truth_img = cv2.imread(img_file)
    # truth_img = cv2.resize(truth_img, (480,640), interpolation=cv2.INTER_AREA)

    img = np.copy(truth_img)
    yolo_img = np.copy(truth_img)

    detections = yolo.detect(yolo_img)

    object_3d_results = []

    for detection in detections:
        
        print(detection.detected_class)
        if not averages.recognized_class(detection.detected_class):
            continue

        # this is throwing when the 2d bbox is invalid
        # TODO: better check
        detectedObject = DetectedObject(img, detection.detected_class, detection.box_2d, calib_file)

        theta_ray = detectedObject.theta_ray
        input_img = detectedObject.img
        proj_matrix = detectedObject.proj_matrix
        box_2d = detection.box_2d
        detected_class = detection.detected_class

        input_tensor = torch.zeros([1,3,224,224])
        input_tensor[0,:,:,:] = input_img

        [orient, conf, dim] = model(input_tensor)
        orient = orient.cpu().data.numpy()[0, :, :]
        conf = conf.cpu().data.numpy()[0, :]
        dim = dim.cpu().data.numpy()[0, :]

        dim += averages.get_item(detected_class)
        argmax = np.argmax(conf)
        orient = orient[argmax, :]
        cos = orient[0]
        sin = orient[1]
        alpha = np.arctan2(sin, cos)
        alpha += angle_bins[argmax]
        alpha -= np.pi

        location, corners = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img)

        object_dims = object_info(corners)
        print("OBJ: ",object_dims)
        object_3d_results.append(object_dims+[detection.detected_class])
        # location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray)
    return object_3d_results
    # if show_yolo:
    #     numpy_vertical = np.concatenate((truth_img, img), axis=1)
    #     cv2.imshow('SPACE for next image, any other key to exit', numpy_vertical)
    #     # cv2.imwrite("out_"+img_id, numpy_vertical)
    # else:
    #     img = cv2.resize(img, (540,1160))
    #     cv2.imshow('3D detections', img)
    #     # cv2.imwrite("out_"+img_id, img)
    # print("\n")
    # print('-------------')

    
    # if cv2.waitKey(0) != 32: # space bar
    #     exit()
    

