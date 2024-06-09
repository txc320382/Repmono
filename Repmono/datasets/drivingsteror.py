import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class DrivingStereoDataset(Dataset):
    def __init__(self, data_path, filenames, height, width, frame_idxs, num_scales, is_train=False, img_ext='.jpg'):
        super(DrivingStereoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = cv2.INTER_LINEAR
        self.frame_idxs = frame_idxs
        self.is_train = is_train
        self.img_ext = img_ext

        self.full_res_shape = (self.width, self.height)
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = cv2.resize

    def preprocess(self, inputs, color_aug):
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i)],
                                                        (self.width // (2 ** i), self.height // (2 ** i)))
                    inputs[(n + "_aug", im, i)] = color_aug(inputs[(n, im, i)])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}

        do_color_aug = self.is_train and np.random.random() > 0.5
        do_flip = self.is_train and np.random.random() > 0.5

        line = self.filenames[index].strip().split()
        folder = line[0]
        frame_index = int(line[1])

        inputs[("color", 0, -1)] = self.get_color(folder, frame_index, do_flip)

        if do_color_aug:
            color_aug = self.brightness_contrast
        else:
            color_aug = lambda x: x

        self.preprocess(inputs, color_aug)

        return inputs

    def get_color(self, folder, frame_index, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index))
        if do_flip:
            color = np.fliplr(color)
        return color

    def get_image_path(self, folder, frame_index):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, folder, "left-image-half-size", f_str)
        return image_path

    def loader(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def brightness_contrast(self, img):
        brightness = np.random.uniform(0.8, 1.2)
        contrast = np.random.uniform(0.8, 1.2)
        return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)