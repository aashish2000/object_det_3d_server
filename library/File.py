"""
Functions to read from files
TODO: move the functions that read label from Dataset into here
"""
import numpy as np
from matrix import pose_to_projection
from matrix import projection_mat


def get_calibration_cam_to_image(cab_f):
    # return(projection_mat(cab_f))
    # for line in open(cab_f):
    #     if 'P2:' in line:
    #         cam_to_img = line.strip().split(' ')
    #         cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
    #         cam_to_img = np.reshape(cam_to_img, (3, 4))
    #         return cam_to_img

    file_not_found(cab_f)

def get_P(cab_f):
    thresh = np.array([[4.60271999e+01, 3.50817969e+02, 5.48055634e+02, 5.58621832e+04],
       [8.69255302e+01, 1.05815036e+04, 5.76459932e+03, 1.13210564e+05],
       [6.43141932e-02, 3.48978506e-01, 1.87972410e+00, 1.89642512e+02]])

    return(projection_mat(cab_f))
    # return(pose_to_projection(cab_f))
    # line = "P_rect_02: 4.28580583e+03 1.20069357e+02 2.09690272e+02 -9.26269304e+01 2.32524187e+01 4.26259533e+03 2.12167314e+03 -5.47921160e+01 -9.57045238e-03 1.09078176e-01 9.93987083e-01 1.22562423e-02"
    # line = "P_rect_02: 5.74446419e+03 3.75876599e+02 4.24484398e+02 5.58431094e+04 -1.25080230e+02 6.27961040e+03 1.17683222e+03 1.13180841e+05 -4.90249773e-03 4.84427826e-01 8.91360539e-01 1.89657742e+02"
    # # # for line in open(cab_f):
    # if 'P_rect_02' in line:
    #     cam_P = line.strip().split(' ')
    #     cam_P = np.asarray([float(cam_P) for cam_P in cam_P[1:]])
    #     return_matrix = np.zeros((3,4))
    #     return_matrix = cam_P.reshape((3,4))
    #     return return_matrix

    # # try other type of file
    # return get_calibration_cam_to_image

def get_R0(cab_f):
    for line in open(cab_f):
        if 'R0_rect:' in line:
            R0 = line.strip().split(' ')
            R0 = np.asarray([float(number) for number in R0[1:]])
            R0 = np.reshape(R0, (3, 3))

            R0_rect = np.zeros([4,4])
            R0_rect[3,3] = 1
            R0_rect[:3,:3] = R0

            return R0_rect

def get_tr_to_velo(cab_f):
    for line in open(cab_f):
        if 'Tr_velo_to_cam:' in line:
            Tr = line.strip().split(' ')
            Tr = np.asarray([float(number) for number in Tr[1:]])
            Tr = np.reshape(Tr, (3, 4))

            Tr_to_velo = np.zeros([4,4])
            Tr_to_velo[3,3] = 1
            Tr_to_velo[:3,:4] = Tr

            return Tr_to_velo

def file_not_found(filename):
    print("\nError! Can't read calibration file, does %s exist?"%filename)
    exit()
