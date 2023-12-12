import argparse
import os, sys
from glob import glob
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
import albumentations as albu
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    img = cv2.imread("images/out_112.bmp")
    h, w, _ = img.shape

    mask = np.zeros_like(img)
    augmented = val_transform(image=img, mask=mask)
    img = augmented['image']
    # mask = augmented['mask']
    img = img.astype('float32') / 255
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img)

    with torch.no_grad():
        input = img.unsqueeze(0)
        input = input.cuda()

        # compute output
        if config['deep_supervision']:
            output = model(input)[-1]
        else:
            output = model(input)

        output = torch.sigmoid(output).cpu().numpy()
        output = (cv2.resize(output[0, 0], dsize=(w, h)) * 255).astype('uint8')
        cv2.imwrite("images/pred_mask_112.bmp", output)
        print("image saved at: images/pred_mask_112.bmp")

        overlay = cv2.imread("images/out_112.bmp")

        for i in range(h):
            for j in range(w):
                if output[i, j]:
                    overlay[i, j] = [0, 255, 255]

        cv2.imwrite("images/overlay_112.bmp", overlay.astype('uint8'))
        print("image saved at: images/overlay_112.bmp")

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
