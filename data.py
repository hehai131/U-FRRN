from __future__ import print_function

import glob
import os
import random

import torch.utils.data as datautils
import torchvision.datasets as visdsets
import torchvision.transforms as transforms
from PIL import Image


def vis_subdir(path, fmt):
    # print(path)
    assert fmt in ('jpg', 'png')
    if fmt == 'jpg':
        files = glob.glob(os.path.join(path, '*.jpg'))
    elif fmt == 'png':
        files = glob.glob(os.path.join(path, '*.png'))
    # print(os.listdir(path))
    for i in os.listdir(path):
        sub_path = path + '/' + i
        if os.path.isdir(sub_path):
            files = files + vis_subdir(sub_path, fmt)
    return files

def joint_vis_subdir(path, fmt):
    # print(path)
    assert fmt in ('jpg', 'png')
    if fmt == 'jpg':
        files_m = glob.glob(os.path.join(path, 'm.jpg'))
        files_g = glob.glob(os.path.join(path, 'g.jpg'))
        files_r = glob.glob(os.path.join(path, 'r.jpg'))
    elif fmt == 'png':
        files_m = glob.glob(os.path.join(path, '*-m-*.png')) # real
        files_g = glob.glob(os.path.join(path, '*-g-*.png')) # background
        files_r = glob.glob(os.path.join(path, '*-r-*.png')) # reflection
    # print(os.listdir(path))
    for i in os.listdir(path):
        sub_path = path + '/' + i
        if os.path.isdir(sub_path):
            m, g, r = joint_vis_subdir(sub_path, fmt)
            files_m = files_m + m
            files_g = files_g + g
            files_r = files_r + r
    return files_m, files_g, files_r


def preprocess(image, transform):
    scale_size = transform['scale_size']
    crop_size = transform['crop_size']
    horizontal_flip = transform['horizontal_flip']

    assert scale_size >= crop_size
    # scale
    scale = transforms.Resize(scale_size)
    image = scale(image)
    width, height = image.size

    # crop
    if scale_size > crop_size:
        x = random.randint(0, scale_size - crop_size)
        y = random.randint(0, scale_size - crop_size)
        region = (x, y, x + crop_size, y + crop_size)
        image = image.crop(region)

    # flip
    if horizontal_flip and random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    return image

def joint_preprocess(image1, image2, image3, transform):
    scale_size = transform['scale_size']
    crop_size = transform['crop_size']
    horizontal_flip = transform['horizontal_flip']

    assert scale_size >= crop_size
    # scale
    scale = transforms.Resize(scale_size)
    image1 = scale(image1)
    image2 = scale(image2)
    image3 = scale(image3)
    width, height = image1.size

    # crop
    if scale_size > crop_size:
        x = random.randint(0, scale_size - crop_size)
        y = random.randint(0, scale_size - crop_size)
        region = (x, y, x + crop_size, y + crop_size)
        image1 = image1.crop(region)
        image2 = image2.crop(region)
        image3 = image3.crop(region)

    # flip
    if horizontal_flip and random.random() < 0.5:
        image1 = image1.transpose(Image.FLIP_LEFT_RIGHT)
        image2 = image2.transpose(Image.FLIP_LEFT_RIGHT)
        image3 = image3.transpose(Image.FLIP_LEFT_RIGHT)

    return image1, image2, image3


class ProxyDataset(datautils.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        index = self.indices[index]
        return self.dataset[index]

    def __len__(self):
        return len(self.indices)


class MixedDataset(datautils.Dataset):
    def __init__(self, datasets, min_length, equalize=False):
        super(MixedDataset, self).__init__()

        if equalize:
            self.datasets = []

            # print([len(dset) for dset in datasets])
            max_length = max([len(dset) for dset in datasets])
            max_length = max(max_length, min_length)

            for dset in datasets:
                dset_length = len(dset)
                # print(dset_length)
                if dset_length == max_length:
                    self.datasets.append(dset)
                else:
                    indices = list(range(dset_length))
                    aug_indices = indices * (max_length // dset_length)

                    tail_length = max_length - len(aug_indices)
                    tail_indices = list(indices)
                    random.shuffle(tail_indices)
                    aug_indices += tail_indices[:tail_length]

                    aug_dataset = ProxyDataset(dset, aug_indices)
                    self.datasets.append(aug_dataset)
        else:
            self.datasets = datasets

        self.length = sum([len(dset) for dset in self.datasets])

    def __getitem__(self, index):
        for dset in self.datasets:
            if index < len(dset):
                return dset[index]
            else:
                index -= len(dset)

    def __len__(self):
        return self.length


def split_dataset(dataset, train_factor):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    pivot = int(len(indices) * train_factor)

    train_dataset = ProxyDataset(dataset, indices[:pivot])
    val_dataset = ProxyDataset(dataset, indices[pivot:])

    return train_dataset, val_dataset


class Foreground_Dataset(datautils.Dataset):
    def __init__(self, root, data_sources, transform=None):
        assert data_sources in ('voc')
        self.data_sources = data_sources

        if data_sources == 'voc':
            self.image_paths = vis_subdir(os.path.join(root, 'Images'), 'jpg')

        self.length = len(self.image_paths)
        print(data_sources, self.length)
        self.transform = transform
        self.loader = visdsets.folder.default_loader
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = self.loader(image_path)

        if self.transform is not None:
            image = preprocess(image, self.transform)

        image_tensor = self.to_tensor(image)
        return image_tensor

    def __len__(self):
        return self.length


def get_voc_datasets(voc_root, train_factor, transform):
    dataset = Foreground_Dataset(voc_root, 'voc', transform=transform)
    train_dataset, val_dataset = split_dataset(dataset, train_factor)
    return train_dataset, val_dataset


class Background_Dataset(datautils.Dataset):
    def __init__(self, root, data_sources, transform=None):
        assert data_sources in ('sun')
        self.data_sources = data_sources

        if data_sources == 'sun':
            self.image_paths = vis_subdir(os.path.join(root, 'Images'), 'jpg')

        self.length = len(self.image_paths)
        print(data_sources, self.length)
        self.transform = transform
        self.loader = visdsets.folder.default_loader
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        # print(image_path)
        image = self.loader(image_path)

        if self.transform is not None:
            image = preprocess(image, self.transform)

        image_tensor = self.to_tensor(image)
        return image_tensor

    def __len__(self):
        return self.length


def get_sun_datasets(sun_root, train_factor, transform):
    dataset = Background_Dataset(sun_root, 'sun', transform=transform)
    train_dataset, val_dataset = split_dataset(dataset, train_factor)
    return train_dataset, val_dataset


class RealImage_Dataset(datautils.Dataset):
    def __init__(self, root, data_sources, transform=None):
        assert data_sources in ('real')
        self.data_sources = data_sources

        if data_sources == 'real':
            self.image_paths = vis_subdir(root, 'jpg')
            self.image_paths = self.image_paths + vis_subdir(root, 'png')

        self.length = len(self.image_paths)
        print(data_sources, self.length)
        self.transform = transform
        self.loader = visdsets.folder.default_loader
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = self.loader(image_path)

        if self.transform is not None:
            image = preprocess(image, self.transform)

        image_tensor = self.to_tensor(image)
        return image_tensor

    def __len__(self):
        return self.length


def get_real_datasets(real_root, transform):
    train_path = real_root  # os.path.join(real_root, 'training')
    test_path = real_root  # os.path.join(real_root, 'testing')
    train_dataset = RealImage_Dataset(train_path, 'real', transform=transform)
    val_dataset = RealImage_Dataset(test_path, 'real', transform=transform)
    return train_dataset, val_dataset

#########################################
# val data
class Val_Dataset(datautils.Dataset):
    def __init__(self, root, data_sources, transform=None):
        assert data_sources in ('val')
        self.data_sources = data_sources

        if data_sources == 'val':
            self.image_paths = vis_subdir(root, 'jpg')
            self.image_paths = self.image_paths + vis_subdir(root, 'png')
            self.image_paths.sort()

        self.length = len(self.image_paths)
        print(data_sources, self.length)
        self.transform = transform
        self.loader = visdsets.folder.default_loader
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = self.loader(image_path)

        if self.transform is not None:
            image = preprocess(image, self.transform)

        image_tensor = self.to_tensor(image)
        return image_tensor

    def __len__(self):
        return self.length


def get_val_datasets(real_root, transform = None):
    test_path = real_root
    val_dataset = Val_Dataset(test_path, 'val', transform=transform)
    return val_dataset

#########################################
# Depth data
# class NYU_Depth_Dataset(datautils.Dataset):
#     def __init__(self, root, data_sources, transform=None):
#         assert data_sources in ('depth')
#         self.data_sources = data_sources
#
#         if data_sources == 'depth':
#             self.RGB_image_paths = vis_subdir(os.path.join(root, 'NYU_image'), 'png')
#             self.RGB_image_paths.sort()
#             self.Depth_image_paths = vis_subdir(os.path.join(root, 'NYU_depth'), 'png')
#             self.Depth_image_paths.sort()
#
#         self.length = len(self.RGB_image_paths)
#         print(data_sources, self.length)
#         self.transform = transform
#         self.loader = visdsets.folder.default_loader
#         self.to_tensor = transforms.ToTensor()
#
#     def __getitem__(self, index):
#         rgb_image_path = self.RGB_image_paths[index]
#         depth_image_path = self.Depth_image_paths[index]
#         rgb_image = self.loader(rgb_image_path)
#         depth_image = self.loader(depth_image_path)
#
#         # # deal with image and depth
#         # rgb_image = np.asarray(rgb_image)
#         # depth_image = np.asarray(depth_image)
#         # depth_image = (255-depth_image)/255.0 # normalization
#         # image = rgb_image*depth_image
#         # image = np.uint8(image)
#         # image = Image.fromarray(image)
#
#         if self.transform is not None:
#             rgb_image, depth_image = joint_preprocess(rgb_image, depth_image, self.transform)
#
#         image_tensor = self.to_tensor(rgb_image)
#         mask_tensor = self.to_tensor(depth_image)
#         return image_tensor, mask_tensor
#
#     def __len__(self):
#         return self.length


# def get_NYU_datasets(real_root, transform = None):
#     train_path = real_root
#     test_path = real_root
#     train_dataset = NYU_Depth_Dataset(train_path, 'depth', transform=transform)
#     test_dataset = NYU_Depth_Dataset(test_path, 'depth', transform=transform)
#     return train_dataset, test_dataset

#########################################
# SIRR data
class SIRR_Dataset(datautils.Dataset):
    def __init__(self, root, data_sources, transform=None):
        assert data_sources in ('sirr')
        self.data_sources = data_sources

        if data_sources == 'sirr':
            m_png, g_png, r_png = joint_vis_subdir(root, 'png')
            m_jpg, g_jpg, r_jpg = joint_vis_subdir(root, 'jpg')
            self.m_image_paths = m_png + m_jpg
            self.g_image_paths = g_png + g_jpg
            self.r_image_paths = r_png + r_jpg
            self.m_image_paths.sort()
            self.g_image_paths.sort()
            self.r_image_paths.sort()

            # print(self.m_image_paths)
            # print(self.g_image_paths)
            # print(self.r_image_paths)

        self.length = len(self.m_image_paths)
        print(data_sources, len(self.m_image_paths), len(self.g_image_paths), len(self.r_image_paths))
        self.transform = transform
        self.loader = visdsets.folder.default_loader
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        m_image_path = self.m_image_paths[index]
        g_image_path = self.g_image_paths[index]
        r_image_path = self.r_image_paths[index]
        m_image = self.loader(m_image_path)
        g_image = self.loader(g_image_path)
        r_image = self.loader(r_image_path)

        if self.transform is not None:
            m_image, g_image, r_image = joint_preprocess(m_image, g_image, r_image, self.transform)

        real_tensor = self.to_tensor(m_image)
        background_tensor = self.to_tensor(g_image)
        reflection_tensor = self.to_tensor(r_image)
        return real_tensor, background_tensor, reflection_tensor

    def __len__(self):
        return self.length


def get_SIRR_datasets(sirr_root, transform = None):
    # train_path = os.path.join(sirr_root, 'train')
    # val_path = os.path.join(sirr_root, 'val')
    # train_dataset = SIRR_Dataset(train_path, 'sirr', transform=transform)
    # val_dataset = SIRR_Dataset(val_path, 'sirr', transform=None)
    train_dataset = SIRR_Dataset(sirr_root, 'sirr', transform=transform)
    val_dataset = SIRR_Dataset(sirr_root, 'sirr', transform=None)
    return train_dataset, val_dataset
