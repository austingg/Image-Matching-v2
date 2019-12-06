import os
import pickle

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from config import im_size, device
from utils import ensure_folder

src_folder = '../Image-Matching/data/data/frame/cron20190326/'
dst_folder = 'data/cron20190326_aligned/'
pickle_file = 'data/data.pkl'

if __name__ == "__main__":
    # checkpoint = 'BEST_checkpoint.tar'
    # checkpoint = torch.load(checkpoint)
    # model = checkpoint['model'].module
    # model = model.to(torch.device('cpu'))
    # model.eval()

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
            img = cv.resize(raw, (im_size, im_size))
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

            # output = output * im_size
            # print('output: ' + str(output))
            # print('output.shape: ' + str(output.shape))

            # img = draw_bboxes(raw, output, size=15)
            # cv.imwrite('test/result.jpg', img)

            src_pts = output
            dst_pts = np.float32([[0, 0], [0, im_size], [im_size, im_size], [im_size, 0]]).reshape(-1, 1, 2)
            M, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            # print(M)

            img = cv.warpPerspective(raw, M, (im_size, im_size), cv.INTER_CUBIC)
            cv.imwrite(dst_path, img)

        else:
            img = cv.resize(raw, (im_size, im_size), cv.INTER_CUBIC)
            cv.imwrite(dst_path, img)
