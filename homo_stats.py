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
            dst_pts = np.float32([[0, 0], [0, dst_im_size], [dst_im_size, dst_im_size], [dst_im_size, 0]]).reshape(-1, 1, 2)
            M, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            print(M)
            break

