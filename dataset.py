import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch.utils.data as data
from PIL import Image
import pickle
import random
import os
import os.path
import numpy as np
import cv2

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, extensions):
    # self.label=
    # real_label = dir.replace('\n', '').split(' ')[-1]
    # image_path = dir.replace('\n', '').split(' ')[0]
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                item = (path, 0)
                images.append(item)

    return images



class DatasetFolder(data.Dataset):
    def __init__(self, root, loader, extensions, vocab_path, dict_path, transform=None, target_transform=None):

        # ------ baseline code ---------------
        self.root = root
        self.loader = loader
        self.extensions = extensions
        # self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

        # 改动 ---------- 6.13 -----------------
        # real_labels = make_dataset(root, extensions)
        real_labels = open(root, encoding='UTF-8').readlines()

        with open(dict_path, 'rb') as f:
            label_sequence = pickle.load(f)  # 打开标签序列
        vocab = open(vocab_path, encoding='UTF-8').readlines()[0]
        radical_vocab_size = len(vocab)  # 笔画长度
        # cal the max len of all captions
        max_len = 0
        for v in label_sequence.values():
            if len(v) > max_len:
                max_len = len(v)

        label_max_len = max_len + 2

        # random.shuffle(real_labels)

        self.labels = real_labels

        self.label_sequence = label_sequence
        self.radical_vocab_size = radical_vocab_size
        self.label_max_len = label_max_len

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        label = self.labels[index]
        real_label = label.replace('\n', '').split(' ')[-1]
        image_path = label.replace('\n', '').split(' ')[0]
        sample = self.loader(image_path)
        # image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            # image = self.transform(image)
            image = self.transform(sample)


        # process label
        label = self.label_sequence[real_label]
        # labels_len = actual label length + <start> + <end> =  len(label) + 2
        label_len = len(label) + 2
        # padding label with <start>, <end> and <pad>

        label = [int(c) for c in label]
        # add <start> : 473  apyt begining
        label = [self.radical_vocab_size] + label
        # add <end> : 474 at end
        label = label + [self.radical_vocab_size + 1]
        # add <pad> : 475
        while len(label) < self.label_max_len:
            label.append(self.radical_vocab_size + 2)

        return image, np.array(label), np.array(label_len), image_path

    def __len__(self):
        # return len(self.samples)
        return len(self.labels)

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



def pil_loader(image_path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(image_path, 'rb') as f:
        img = Image.open(f)
        # return img.convert('L')
        return img.convert('RGB')


def default_loader(image_path):
    return pil_loader(image_path)


class ImageFolder(DatasetFolder):
    # def __init__(self, root, transform=None, target_transform=None,
    #              loader=default_loader):
    def __init__(self, root, vocab_path, dict_path, transform=None, target_transform=None,
                  loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS, vocab_path, dict_path,
                                          transform=transform,
                                          target_transform=target_transform)
        # self.imgs = self.samples

