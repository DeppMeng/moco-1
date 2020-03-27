# ------------------------------------------------------------------------------
# classification.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Depu Meng (mdp@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import logging
import zipfile

import torch.utils.data as data
import torch.distributed as dist

from PIL import Image
import cv2
import numpy as np

# import moco.utils.zip as phillyzip


logger = logging.getLogger(__name__)


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(dirs):
    classes = [d for d in dirs]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def get_zip_info(zip_file):
    dirs = []
    files = []

    all_namelist = zip_file.namelist()
    for name in all_namelist:
        if name.endswith('/'):
            dirs.append(name.strip('/'))
        else:
            files.append(name)

    return dirs, files


def make_dist_dataset(zip_file_name, extensions):
    dirs = []
    files = []
    images = []
    classes = []
    class_to_idx = {}
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    with zipfile.ZipFile(zip_file_name, 'r') as zip_file:
        dirs, files = get_zip_info(zip_file)
        classes, class_to_idx = find_classes(dirs)

        dataset_index_list = list(range(len(files)))
        subset_index_list = dataset_index_list[rank:len(files):world_size]
        for idx, f in enumerate(files):
            dir = f.split('/')[0]
            if dir in classes and is_image_file(f) and idx in subset_index_list:
                # print(zip_file.read(f))
                images.append((zip_file.read(f), class_to_idx[dir]))

    return classes, class_to_idx, images


class DistCachedZipFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        zip_file@/class_x/xxx.ext
        zip_file@/class_x/xxy.ext
        zip_file@/class_x/xxz.ext

        zip_file@/class_y/123.ext
        zip_file@/class_y/nsdf3.ext
        zip_file@/class_y/asd932_.ext

    Args:
        zip_file (string): Zip file path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, zip_file_name, loader, extensions, transform=None, target_transform=None):
        classes, class_to_idx, samples = make_dist_dataset(zip_file_name, extensions)
        logger.info('=> {} classes are added'.format(len(classes)))
        logger.info('=> {} samples are added'.format(len(samples)))

        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + zip_file_name + "\n"))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        buffer, target = self.samples[index]
        sample = self.loader(io.BytesIO(buffer))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(file):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    img = Image.open(file)
    return img.convert('RGB')


# def cv2_loader(data):
#     img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return Image.fromarray(img)


def default_loader(file):
    return pil_loader(file)


class DistImageZipFolder(DistCachedZipFolder):
    """A generic data loader where the images are arranged in this way: ::
        zip_file@/class_x/xxx.ext
        zip_file@/class_x/xxy.ext
        zip_file@/class_x/xxz.ext

        zip_file@/class_y/123.ext
        zip_file@/class_y/nsdf3.ext
        zip_file@/class_y/asd932_.ext

    Args:
        zip_file_name (string): Zip file path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, zip_file_name, transform=None, target_transform=None,
                 loader=default_loader):
        super(DistImageZipFolder, self).__init__(zip_file_name, loader, IMG_EXTENSIONS,
                                        transform=transform,
                                        target_transform=target_transform)
        self.imgs = self.samples
