import math
import os

import numpy as np
import cv2


def get_poses(data_dir):
    with open(data_dir) as f:
        poses = np.array([[float(x) for x in line.split()] for line in f])
    return poses


def p_to_se3(p):
    SE3 = np.array([
        [p[0], p[1], p[2], p[3]],
        [p[4], p[5], p[6], p[7]],
        [p[8], p[9], p[10], p[11]],
        [0, 0, 0, 1]
    ])
    return SE3


def get_ground_6d_poses(p, p2):
    SE1 = p_to_se3(p)
    SE2 = p_to_se3(p2)

    SE12 = np.matmul(np.linalg.inv(SE1), SE2)

    pos = np.array([SE12[0][3], SE12[1][3], SE12[2][3]])
    angles = rotation_matrix_to_euler_angles(SE12[:3, :3])
    return np.concatenate((angles, pos))  


def get_ground_6d_poses_quat(p, p2):
    SE1 = p_to_se3(p)
    SE2 = p_to_se3(p2)

    SE12 = np.matmul(np.linalg.inv(SE1), SE2)

    pos = np.array([SE12[0][3], SE12[1][3], SE12[2][3]])
    quat = rotation_matrix_to_quaternion(SE12[:3, :3])
    return np.concatenate((quat, pos)) 


def is_rotation_matrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotation_matrix_to_euler_angles(R):   
    assert(is_rotation_matrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def rotation_matrix_to_quaternion(R):
    assert (is_rotation_matrix(R))

    qw = np.sqrt(1 + np.sum(np.diag(R))) / 2.0
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)

    return np.array([qw, qx, qy, qz])


def load_dof6_labels(pose_path):
    pose = get_poses(pose_path)
    dof6_labels = []
    
    for i in range(1,len(pose)):
        dof6 = get_ground_6d_poses(pose[i-1],pose[i])
        dof6_labels.append(dof6)
    return dof6_labels   

def euler_to_rotation_matrix(theta) :
     
    R_x = np.array([[1, 0,                  0                  ],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0]) ]
                    ])
         
         
                     
    R_y = np.array([[math.cos(theta[1]),  0, math.sin(theta[1])  ],
                    [0,                   1,  0                  ],
                    [-math.sin(theta[1]), 0,  math.cos(theta[1]) ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]),  0],
                    [0,                  0,                   1]
                    ])
                     
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R

def dof6_to_eval_path(dof):
    SE3 = []
    dof = np.array(dof)
    for i in range(len(dof)):
        rotm = euler_to_rotation_matrix(dof[i,:3])
        SE3.append(np.array([[rotm[0,0], rotm[0,1], rotm[0,2], dof[i,3]],
                            [rotm[1,0], rotm[1,1], rotm[1,2], dof[i,4]],
                            [rotm[2,0], rotm[2,1], rotm[2,2], dof[i,5]],
                            [0, 0, 0, 1]]))
    SE3 = np.array(SE3)
    path = np.zeros((3,len(SE3)))
    for i in range(1,len(SE3)):
        SE3[i] = np.matmul(SE3[i-1,:,:],SE3[i,:,:])
    path[0] = SE3[:,0,3]
    path[1] = SE3[:,1,3]
    path[2] = SE3[:,2,3]
    return path