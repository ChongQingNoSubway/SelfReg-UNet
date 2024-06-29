import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from batchgenerators.augmentations.spatial_transformations import augment_spatial_2
# from torchvision.transforms import v2
from torchvision.transforms import functional as VF


def dino_augmentation(image,label):
    #print(label.shape)
    #print(image.shape)
    patch_shape = label.shape


    image = image.reshape(1,1,512,512)
    label = label.reshape(1,1,512,512)

    image, label = augment_spatial_2(image,label,patch_size=patch_shape,do_elastic_deform=True,do_rotation=False,do_scale=False,random_crop=False)

    image = image.reshape(512,512)
    label = label.reshape(512,512)

    return image,label

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def get_random_elastic_params(alpha, sigma, size):
    # _, h, w = VF.get_dimensions(image)
    # size = [h, w]
    alpha = 20.0
    sigma = 5.0
    dx = torch.rand([1, 1] + size) * 2 -1
    if sigma > 0:
        kx = int(8 * sigma + 1)
        if kx % 2 == 0:
            kx += 1
        dx = VF.gaussian_blur(dx, [kx, kx], sigma)
    dx = dx * alpha / size[0]

    dy = torch.rand([1, 1] + size) * 2 -1
    if sigma > 0:
        ky = int(8 * sigma + 1)
        if ky % 2 == 0:
            ky += 1
        dy = VF.gaussian_blur(dy, [ky, ky], sigma)
    dy = dy * alpha / size[1]
    return torch.concat([dx, dy], 1).permute([0, 2, 3, 1])

# def elastic


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class RandomGenerator_DINO(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        #elif random.random() > 0.5:
            #image,label = dino_augmentation(image,label)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))

        # if random.random() > 0.5:
        #     image = v2.ElasticTransform()(image)

        sample = {'image': image, 'label': label.long()}
        return sample


class RandomGenerator_DINO_Deform(object):
    def __init__(self, output_size,alpha=20.,sigma=5.):
        self.output_size = output_size
        self.alpha = 20.0
        self.sigma = 5.0

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        #elif random.random() > 0.5:
            #image,label = dino_augmentation(image,label)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0) # Bx1x224x224
        label = torch.from_numpy(label.astype(np.float32)) # Bx224x224

        # if random.random() > 0.5:
        _, h, w = VF.get_dimensions(image)
        size = [int(h), int(w)]
        # print(size)
        displacement = get_random_elastic_params(self.alpha, self.sigma, size)
        # print(displacement.shape)
        image_dino = VF.elastic_transform(image, displacement, VF.InterpolationMode.BILINEAR, 0)
        # label_dino = VF.elastic_transform(label, displacement, VF.InterpolationMode.NEAREST, 0)
        # image 
        sample = {'image': image, 'label': label.long(), 'image_dino': image_dino, 'disp':displacement}
        return sample

class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None,transform_dino=None):
        self.transform = transform  # using transform in torch!
        self.transform_dino = transform_dino
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            # dino_image,dino_label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
            # dino_image,dino_label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        # dino_sample = {'image': dino_image, 'label':dino_label}

        if self.transform:
            sample = self.transform(sample)

            # dino_sample = self.transform_dino(dino_sample)
            # dino_sample = self.transform_dino(dino_sample)

        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample#,dino_sample
