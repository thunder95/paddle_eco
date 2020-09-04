#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import cv2
import math
import random
import functools

try:
    import cPickle as pickle
    from cStringIO import StringIO
except ImportError:
    import pickle
    from io import BytesIO
import numpy as np
import paddle
from PIL import Image
import logging
from numpy.random import randint

logger = logging.getLogger(__name__)
python_ver = sys.version_info

"""
num_segments: 默认是3， kinetics设置的是16， 复现需要32
reader的逻辑:
1. 先打开list文件，读取各个pkl， 其中pkl是单个视频的信息记录，函数reader()
2. 对单个pkl进行解析，从中获取帧和类别信息, 函数decode_pickle()
3. 执行提取图片的逻辑video_loader, 修改时对应pytorch函数：
    - 针对训练集，测试集和验证集才去不同的采样策略
4. img_transforms对图像进行变换， 并生成 batch*32*3*224*224
"""


class KineticsReader(object):
    """
    Data reader for kinetics dataset of two format mp4 and pkl.
    1. mp4, the original format of kinetics400
    2. pkl, the mp4 was decoded previously and stored as pkl
    In both case, load the data, and then get the frame data in the form of numpy and label as an integer.
     dataset cfg: format
                  num_classes
                  seg_num
                  short_size
                  target_size
                  num_reader_threads
                  buf_size
                  image_mean
                  image_std
                  batch_size
                  list
    """

    def __init__(self, name, mode, cfg):
        self.cfg = cfg
        self.mode = mode
        self.name = name
        self.format = cfg.MODEL.format
        self.num_classes = self.get_config_from_sec('model', 'num_classes')
        self.seg_num = self.get_config_from_sec('model', 'seg_num') #pytorch: num_segments
        self.seglen = self.get_config_from_sec('model', 'seglen')

        self.new_length = self.get_config_from_sec('model', 'new_length') #pytorch: new_length
        self.modality = self.get_config_from_sec('model', 'modality') #pytorch: modality
        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self.seg_num = self.get_config_from_sec(mode, 'seg_num', self.seg_num)
        self.short_size = self.get_config_from_sec(mode, 'short_size')
        self.target_size = self.get_config_from_sec(mode, 'target_size')
        self.num_reader_threads = self.get_config_from_sec(mode,
                                                           'num_reader_threads')
        self.buf_size = self.get_config_from_sec(mode, 'buf_size')
        self.enable_ce = self.get_config_from_sec(mode, 'enable_ce')

        self.img_mean = np.array(cfg.MODEL.image_mean).reshape(
            [3, 1, 1]).astype(np.float32)
        self.img_std = np.array(cfg.MODEL.image_std).reshape(
            [3, 1, 1]).astype(np.float32)
        # set batch size and file list
        self.batch_size = cfg[mode.upper()]['batch_size']
        self.filelist = cfg[mode.upper()]['filelist']
        if self.enable_ce:
            random.seed(0)
            np.random.seed(0)



    def get_config_from_sec(self, sec, item, default=None):
        if sec.upper() not in self.cfg:
            return default
        return self.cfg[sec.upper()].get(item, default)

    def create_reader(self):
        _reader = self._reader_creator(self.filelist, self.mode, seg_num=self.seg_num, seglen=self.seglen,
                                       short_size=self.short_size, target_size=self.target_size,
                                       img_mean=self.img_mean, img_std=self.img_std,
                                       shuffle=(self.mode == 'train'),
                                       num_threads=self.num_reader_threads,
                                       buf_size=self.buf_size, format=self.format)

        def _batch_reader():
            batch_out = []
            for imgs, label in _reader():
                if imgs is None:
                    continue
                batch_out.append((imgs, label))
                if len(batch_out) == self.batch_size:
                    yield batch_out
                    batch_out = []

        return _batch_reader

    def _reader_creator(self,
                        pickle_list,
                        mode,
                        seg_num,
                        seglen,
                        short_size,
                        target_size,
                        img_mean,
                        img_std,
                        shuffle=False,
                        num_threads=1,
                        buf_size=1024,
                        format='pkl'):
        def decode_mp4(sample, mode, seg_num, seglen, short_size, target_size,
                       img_mean, img_std):
            sample = sample[0].split(' ')
            mp4_path = sample[0]
            # when infer, we store vid as label
            label = int(sample[1])
            try:
                imgs = mp4_loader(mp4_path, seg_num, seglen, mode)
                if len(imgs) < 1:
                    logger.error('{} frame length {} less than 1.'.format(
                        mp4_path, len(imgs)))
                    return None, None
            except:
                logger.error('Error when loading {}'.format(mp4_path))
                return None, None

            return imgs_transform(imgs, label, mode, seg_num, seglen, \
                                  short_size, target_size, img_mean, img_std)

        #解析pickle文件
        def decode_pickle(sample, mode, seg_num, seglen, short_size,
                          target_size, img_mean, img_std):
            pickle_path = sample[0]
            try:
                if python_ver < (3, 0):
                    data_loaded = pickle.load(open(pickle_path, 'rb'))
                else:
                    data_loaded = pickle.load(
                        open(pickle_path, 'rb'), encoding='bytes')

                vid, label, frames = data_loaded
                if len(frames) < 1:
                    logger.error('{} frame length {} less than 1.'.format(
                        pickle_path, len(frames)))
                    return None, None
            except:
                logger.info('Error when loading {}'.format(pickle_path))
                return None, None

            if mode == 'train' or mode == 'valid' or mode == 'test':
                ret_label = label
            elif mode == 'infer':
                ret_label = vid

            #加载视频帧
            imgs = video_loader(frames, seg_num, mode, self.new_length) #pytorch: __getitem__
            return imgs_transform(imgs, ret_label, mode, seg_num, seglen, \
                                  short_size, target_size, img_mean, img_std)

        def imgs_transform(imgs, label, mode, seg_num, seglen, short_size,
                           target_size, img_mean, img_std):

            if mode == "train":
                #pytorch: GroupMultiScaleCrop, GroupRandomHorizontalFlip
                #注意pytorch中这里是target_size(224),不是short_size
                imgs = group_multi_scale_crop(imgs, target_size)
                imgs = group_random_flip(imgs)
            else:
                #pytorch: GroupScale(int(scale_size)), GroupCenterCrop(crop_size)
                imgs = group_scale(imgs, 256) #pytorch: 224*256//224
                imgs = group_center_crop(imgs, target_size)
            

            
            #转换通道， 这里不需要除以255
            np_imgs = (np.array(imgs[0]).astype('float32').transpose(
                (2, 0, 1))).reshape(1, 3, target_size, target_size)
            for i in range(len(imgs) - 1):
                img = (np.array(imgs[i + 1]).astype('float32').transpose(
                    (2, 0, 1))).reshape(1, 3, target_size, target_size)
                np_imgs = np.concatenate((np_imgs, img))

            #归一化处理
            imgs = np_imgs
            imgs -= img_mean
            imgs /= img_std

            #输入的大小 segnum*3*224*244
            imgs = np.reshape(imgs, (seg_num, 3, target_size, target_size))

            return imgs, label

        def reader():
            with open(pickle_list) as flist:
                lines = [line.strip() for line in flist]
                if shuffle:
                    random.shuffle(lines)
                for line in lines:
                    pickle_path = line.strip()
                    yield [pickle_path]

        if format == 'pkl':
            decode_func = decode_pickle
        elif format == 'mp4':
            decode_func = decode_mp4
        else:
            raise "Not implemented format {}".format(format)

        mapper = functools.partial(
            decode_func,
            mode=mode,
            seg_num=seg_num,
            seglen=seglen,
            short_size=short_size,
            target_size=target_size,
            img_mean=img_mean,
            img_std=img_std)

        return paddle.reader.xmap_readers(mapper, reader, num_threads, buf_size)

#pytorch: GroupMultiScaleCrop, 代码应该是一致的
def group_multi_scale_crop(img_group, target_size, scales=None, \
                           max_distort=1, fix_crop=True, more_fix_crop=True):
    scales = scales if scales is not None else [1, .875, .75, .66] #pytorch: check
    input_size = [target_size, target_size]

    im_size = img_group[0].size

    # get random crop offset
    def _sample_crop_size(im_size):
        image_w, image_h = im_size[0], im_size[1]

        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in scales]
        crop_h = [
            input_size[1] if abs(x - input_size[1]) < 3 else x
            for x in crop_sizes
        ]
        crop_w = [
            input_size[0] if abs(x - input_size[0]) < 3 else x
            for x in crop_sizes
        ]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_step = (image_w - crop_pair[0]) / 4
            h_step = (image_h - crop_pair[1]) / 4

            ret = list()
            ret.append((0, 0))  # upper left
            if w_step != 0:
                ret.append((4 * w_step, 0))  # upper right
            if h_step != 0:
                ret.append((0, 4 * h_step))  # lower left
            if h_step != 0 and w_step != 0:
                ret.append((4 * w_step, 4 * h_step))  # lower right
            if h_step != 0 or w_step != 0:
                ret.append((2 * w_step, 2 * h_step))  # center

            if more_fix_crop:
                ret.append((0, 2 * h_step))  # center left
                ret.append((4 * w_step, 2 * h_step))  # center right
                ret.append((2 * w_step, 4 * h_step))  # lower center
                ret.append((2 * w_step, 0 * h_step))  # upper center

                ret.append((1 * w_step, 1 * h_step))  # upper left quarter
                ret.append((3 * w_step, 1 * h_step))  # upper right quarter
                ret.append((1 * w_step, 3 * h_step))  # lower left quarter
                ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

            w_offset, h_offset = random.choice(ret)

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    crop_w, crop_h, offset_w, offset_h = _sample_crop_size(im_size)
    crop_img_group = [
        img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        for img in img_group
    ]
    ret_img_group = [
        img.resize((input_size[0], input_size[1]), Image.BILINEAR)
        for img in crop_img_group
    ]

    return ret_img_group


def group_random_crop(img_group, target_size):
    w, h = img_group[0].size
    th, tw = target_size, target_size

    assert (w >= target_size) and (h >= target_size), \
        "image width({}) and height({}) should be larger than crop size".format(w, h, target_size)

    out_images = []
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    for img in img_group:
        if w == tw and h == th:
            out_images.append(img)
        else:
            out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

    return out_images

#pytorch: GroupRandomHorizontalFlip, 代码应该是一致的
def group_random_flip(img_group):
    v = random.random()
    if v < 0.5:
        ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
        return ret
    else:
        return img_group

#pytorch: GroupCenterCrop, 这里长宽不一定都大于224，后面优化
def group_center_crop(img_group, target_size):
    img_crop = []
    for img in img_group:
        w, h = img.size
        th, tw = target_size, target_size
        assert (w >= target_size) and (h >= target_size), \
            "image width({}) and height({}) should be larger than crop size".format(w, h, target_size)
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img_crop.append(img.crop((x1, y1, x1 + tw, y1 + th)))

    return img_crop

#pytorch:这里有变化，原代码resize保持长宽比例，而不是4.0/3.0
def group_scale(imgs, target_size):
    resized_imgs = []
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.size
        if (h>=w):
            resized_imgs.append(img.resize((int(target_size*h/w), target_size), Image.BILINEAR))
        else:
            resized_imgs.append(img.resize((target_size, int(target_size*w/h)), Image.BILINEAR))
    return resized_imgs


def imageloader(buf):
    if isinstance(buf, str):
        img = Image.open(buf)
    else:
        img = Image.open(BytesIO(buf))

    return img.convert('RGB')


#读取视频的一些辅助函数
def _sample_indices(vlen, seglen, new_len):
    average_duration = (vlen - new_len + 1) // seglen
    if average_duration > 0:
        offsets = np.multiply(list(range(seglen)), average_duration) + randint(average_duration, size=seglen)
    elif vlen > seglen:
        offsets = np.sort(randint(vlen - new_len + 1, size=seglen))
    else:
        offsets = np.zeros((seglen,))
    return offsets + 1


def _get_val_indices(vlen, seglen, new_len):
        if vlen >seglen + new_len - 1:
            tick = (vlen - new_len + 1) / float(seglen)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(seglen)])
        else:
            offsets = np.zeros((seglen,))
        return offsets + 1
def _get_test_indices(vlen, seglen, new_len):
        tick = (vlen -new_len + 1) / float(seglen)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(seglen)])
        return offsets + 1


def video_loader(frames, seglen, mode, new_len):
    vlen = len(frames)
    if not mode == "test":
        segment_indices = _sample_indices(vlen, seglen, new_len) if mode=="train" else _get_val_indices(vlen, seglen, new_len)
    else:
        segment_indices = _get_test_indices(vlen, seglen, new_len)
    # print("indices len: ", len(segment_indices))
    imgs = list()
    for seg_ind in segment_indices:
        p = int(seg_ind)
        for i in range(new_len):
            try:
                if p >= vlen:
                    p = vlen-1
                imgs.append(Image.open(frames[p]).convert('RGB'))
            except:
                print("====>img error:   ", vlen, p)
            if p < vlen:
                p += 1
    # print("imgs len:", len(imgs))
    return imgs


#暂未使用
def mp4_loader(filepath, nsample, seglen, mode):
    cap = cv2.VideoCapture(filepath)
    videolen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampledFrames = []
    for i in range(videolen):
        ret, frame = cap.read()
        # maybe first frame is empty
        if ret == False:
            continue
        img = frame[:, :, ::-1]
        sampledFrames.append(img)
    average_dur = int(len(sampledFrames) / nsample)
    imgs = []
    for i in range(nsample):
        idx = 0
        if mode == 'train':
            if average_dur >= seglen:
                idx = random.randint(0, average_dur - seglen)
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i * average_dur
            else:
                idx = i
        else:
            if average_dur >= seglen:
                idx = (average_dur - 1) // 2
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i * average_dur
            else:
                idx = i

        for jj in range(idx, idx + seglen):
            imgbuf = sampledFrames[int(jj % len(sampledFrames))]
            img = Image.fromarray(imgbuf, mode='RGB')
            imgs.append(img)

    return imgs
