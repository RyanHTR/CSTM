import os
import h5py
import pandas as pd
import numpy as np
import torchvision.transforms as transforms

import torch
from torch.utils.data import Dataset
from .word_utils import Corpus
from PIL import Image
import torch.nn.functional as F

from collections import Iterable
from torch.autograd import Variable


class ResizeAnnotation:
    """Resize the largest of the sides of the annotation to a given size"""

    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError("Got inappropriate size arg: {}".format(size))

        self.size = size

    def __call__(self, img):
        im_h, im_w = img.shape[-2:]
        scale_h, scale_w = self.size / im_h, self.size / im_w
        resized_h = int(np.round(im_h * scale_h))
        resized_w = int(np.round(im_w * scale_w))
        out = (
            F.interpolate(
                Variable(img).unsqueeze(0).unsqueeze(0),
                size=(resized_h, resized_w),
                mode="bilinear",
                align_corners=True,
            )
            .squeeze()
            .data
        )
        return out


def load_rgb_frames(image_dir, vid, start, num, margin=1, im_size=512, tv_transform=None):
    frames = []
    for i in range(start, start+num, margin):
        img_path = os.path.join(image_dir, vid, '{:0>5d}.jpg'.format(i))
        img = Image.open(img_path).convert('RGB')
        img = tv_transform(img)  # [3, h, w]
        frames.append(img.numpy())
    frames_array = np.asarray(frames, dtype=np.float32)  # [nf, 3, h, w]
    return torch.from_numpy(frames_array.transpose([1, 0, 2, 3]))


class PairData():
    def __init__(self, pd_data):
        self.frame_path = pd_data[0]
        self.size = pd_data[1:3].astype(int)
        self.instance_id = int(pd_data[3])
        self.video_id, self.frame_id = self.frame_path.split('/')
        self.frame_id = int(self.frame_id)
        self.txt = pd_data[4]


class VideoTextDataset(Dataset):
    def __init__(self, opt, mode='train'):
        super(VideoTextDataset, self).__init__()
        print('loading dataset')
        if mode == 'train':
            fr_name = pd.read_csv('{}/datasets/{}/preprocessed/train.txt'.format(opt.project_root, opt.dataset), header=None).T
            # (n, 2)
        else:
            fr_name = pd.read_csv('{}/datasets/{}/preprocessed/test.txt'.format(opt.project_root, opt.dataset), header=None).T
            # (n, 2)
        # parsing pd data into PairData List
        self._parse_list(fr_name)

        self.corpus = Corpus('{}/word_embedding'.format(opt.project_root))

        # change frame_root here
        self.frame_root = '/mnt/data2/hsf/Code/CVPR2021/A2D/data/Release/frames_cv2'
        self.mask_root = '/mnt/data2/hsf/Code/CVPR2021/A2D/data/a2d_annotation_with_instances'
        self.im_size = opt.resize
        self.opt = opt

        to_tensor = transforms.ToTensor()
        resize = transforms.Resize((opt.resize, opt.resize))

        self.transform = transforms.Compose([resize, to_tensor])
        self.transform_mask = transforms.Compose([ResizeAnnotation(opt.resize)])

    def _parse_list(self, pd_data):
        self.pair_list = []
        for col in pd_data:
            self.pair_list.append(PairData(pd_data[col].values))

    def load_image(self, item, im_size, single=False):
        record = self.pair_list[item]
        video_size = torch.from_numpy(record.size)
        video_id = record.video_id
        frame_id = record.frame_id
        if single:
            frames = load_rgb_frames(self.frame_root, video_id, frame_id, 1, 1, im_size, self.transform)
            return frames, video_size
        start_idx = max(1, frame_id - 8)
        # frames: [3, nf, h, w]
        frames = load_rgb_frames(self.frame_root, video_id, start_idx, 16, self.opt.data_margin, im_size, self.transform)
        return frames, video_size

    def load_mask(self, item):
        record = self.pair_list[item]
        video_id = record.video_id
        frame_id = record.frame_id
        instance_id = record.instance_id
        with h5py.File(os.path.join(self.mask_root, video_id, '{:0>5d}.h5'.format(frame_id)), 'r') as f:
            instances = f['instance'][()]
            idx = np.where(instances==instance_id)[0][0]
            if instances.shape[0] == 1:
                mask = f['reMask'][()].transpose(1, 0)
            else:
                mask = f['reMask'][()][idx].transpose(1, 0)
        mask = self.transform_mask(torch.from_numpy(mask).float())
        mask[mask > 0] = 1
        return mask

    def __getitem__(self, item):
        video, video_size = self.load_image(item, self.im_size, single=self.opt.single_im)
        mask = self.load_mask(item)
        txt, txt_mask = self.corpus.tokenize(self.pair_list[item].txt, self.opt.sentence_length)
        return video_size, video, txt, txt_mask, mask

    def __len__(self):
        return len(self.pair_list)
