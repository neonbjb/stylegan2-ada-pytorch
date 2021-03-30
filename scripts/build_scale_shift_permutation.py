"""A multi-threaded tool that scales and shifts a set of images"""
import os
import os.path as osp
import random
from random import choice

import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data
from tqdm import tqdm
import torch


def main():
    split_img = False
    opt = {}
    opt['n_thread'] = 5
    opt['input_folder'] = ['E:\\4k6k\\datasets\\images\\faces\\flickr_faces_hq']
    opt['save_folder'] = 'E:\\4k6k\\datasets\\images\\faces\\ffhq_256_modded\\shifted'
    opt['shift'] = True
    opt['scale'] = False

    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))

    extract_single(opt)


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.webp', '.WEBP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    """get image path list from image folder"""
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname) and 'ref.jpg' not in fname:
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    if not images:
        print("Warning: {:s} has no valid image file".format(path))
    return images


def get_image_paths(data_type, dataroot, weights=[]):
    """get image path list
    support lmdb or image files"""
    paths, sizes = None, None
    if dataroot is not None:
        if isinstance(dataroot, list):
            paths = []
            for i in range(len(dataroot)):
                r = dataroot[i]
                extends = 1

                # Weights have the effect of repeatedly adding the paths from the given root to the final product.
                if weights:
                    extends = weights[i]
                for j in range(extends):
                    paths.extend(_get_paths_from_images(r))
            paths = sorted(paths)
            sizes = len(paths)
        else:
            paths = sorted(_get_paths_from_images(dataroot))
            sizes = len(paths)
    return paths, sizes


class TiledDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        input_folder = opt['input_folder']
        self.images = get_image_paths('img', input_folder)[0]
        print("Found %i images" % (len(self.images),))

    def __getitem__(self, index):
        return self.get(index)

    def get(self, index):
        path = self.images[index]
        basename = osp.basename(path)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # Greyscale not supported.
        if img is None:
            print("Error with ", path)
            return None
        if len(img.shape) == 2:
            print("Skipping due to greyscale")
            return None

        if self.opt['scale']:
            scale = choice([0,128,256])
        else:
            scale = 0
        h, w, c = img.shape
        shift_x, shift_y = 0,0
        if scale != 256 and self.opt['shift']:
            shift = random.randrange(0,256-scale)
            shift_x = random.randrange(-1,2,2) * shift
            shift_y = random.randrange(-1,2,2) * shift
        img = img[max(shift_y,0)+scale:min(h+shift_y,1024)-scale,max(shift_x,0)+scale:min(w+shift_x,1024)-scale,:]

        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        output_folder = self.opt['save_folder']
        cv2.imwrite(osp.join(output_folder, f'shiftx_{shift_x}_shifty_{shift_y}_scale_{scale}_{basename}'), img)
        return None

    def __len__(self):
        return len(self.images)


def identity(x):
    return x

def extract_single(opt):
    dataset = TiledDataset(opt)
    dataloader = data.DataLoader(dataset, num_workers=opt['n_thread'], collate_fn=identity)
    tq = tqdm(dataloader)
    for spl_imgs in tq:
        pass


if __name__ == '__main__':
    main()
