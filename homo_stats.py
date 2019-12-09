import math
import os
import pickle

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from config import device
from utils import ensure_folder

src_folder = '../Image-Matching/data/data/frame/cron20190326/'
dst_folder = 'data/cron20190326_aligned/'
pickle_file = 'data/data.pkl'
src_im_size = 224
dst_im_size = 256


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

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


# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


if __name__ == "__main__":
    scripted_model_file = 'framedetector_scripted.pt'
    model = torch.jit.load(scripted_model_file)
    model = model.to(device)
    model.eval()

    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    samples = [item for item in data]

    ensure_folder(dst_folder)
    for sample in tqdm(samples):
        dir = sample['dir']
        file = sample['file']
        is_sample = sample['is_sample']
        src_path = '{}{}/{}'.format(src_folder, dir, file)
        raw = cv.imread(src_path)
        dst_path = os.path.join(dst_folder, dir)
        ensure_folder(dst_path)
        dst_path = os.path.join(dst_path, file)

        Tx_list = []
        Ty_list = []
        pitch_list = []
        roll_list = []
        yaw_list = []

        if not is_sample:
            h, w = raw.shape[:2]
            img = cv.resize(raw, (224, 224))
            img = img[..., ::-1]  # RGB
            img = transforms.ToPILImage()(img)
            img = transformer(img)
            img = img.unsqueeze(0)
            img = img.to(device)

            with torch.no_grad():
                outputs = model(img)

            output = outputs[0].cpu().numpy()
            output = np.reshape(output, (4, 1, 2))

            for p in output:
                p[0][0] = p[0][0] * w
                p[0][1] = p[0][1] * h

            src_pts = output
            dst_pts = np.float32([[0, 0], [0, dst_im_size], [dst_im_size, dst_im_size], [dst_im_size, 0]]).reshape(-1,
                                                                                                                   1, 2)
            H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            print('H: ' + str(H))

            data = np.load('calib.npz')
            K = data['mtx']

            num, Rs, Ts, Ns = cv.decomposeHomographyMat(H, K)

            print('Rs: ' + str(Rs))
            # print('Ts: ' + str(Ts))
            # print('Ns: ' + str(Ns))

            angles_list = []
            for i in range(num):
                R = Rs[i]
                if isRotationMatrix(R):
                    angles = rotationMatrixToEulerAngles(R)
                    angles_list.append(angles)

            angles = [ang for ang in angles_list if ang[2] > 0][0]
            R = eulerAnglesToRotationMatrix(angles)
            print('R: ' + str(R))
            break
