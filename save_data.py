import os
import nibabel as nib
from torch.utils.data import Dataset
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils import data
import numpy as np
from skimage.transform import resize
from scipy import ndimage
from tqdm import tqdm
import random
import warnings
import json
import sys
from multiprocessing import Pool

warnings.filterwarnings("ignore", category=UserWarning)
import re

#######################################------Get the parameters------###########################

json_file = sys.argv[1]
print(f"json_file is {json_file}")
with open(json_file, 'r') as f:
    json_file = json.load(f)
folder_path = json_file['folder_path'] + '/'
imgpath_path = folder_path + json_file[
    'imgpath_path'] if 'imgpath_path' in json_file else None
label_path = folder_path + json_file[
    'label_path'] if 'label_path' in json_file else None
img_size = json_file['img_size'] if 'img_size' in json_file else None
out_put_dir = folder_path
task = json_file['task'] if 'task' in json_file else None
data_type = json_file['data_type'] if 'data_type' in json_file else None
number_client = json_file[
    'number_client'] if 'number_client' in json_file else None
number_threads = json_file[
    'number_threads'] if 'number_threads' in json_file else None
number_batch = json_file['number_batch'] if 'number_batch' in json_file else None
rotate = json_file['rotate'] if 'rotate' in json_file else None


#######################################------Start utils functions------###########################
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, id) -> None:
        self.id = id

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        feature_path = folder_path + 'feature'
        label_path = folder_path + 'label'
        if not os.path.exists(feature_path):
            os.mkdir(feature_path)
        if not os.path.exists(label_path):
            os.mkdir(label_path)

        torch.save(image, feature_path + f"/img{self.id}.pth")
        torch.save(label, label_path + f"/label_{self.id}.pth")
        return {'image': image, 'label': label}


def read_nifti_cls_file(filepath, rotate=False):
    print(filepath)
    # Read file
    img = nib.load(filepath)
    # Get raw data H W D C
    img = img.get_fdata().astype(np.float32)
    dim = len(img.shape)
    if dim == 3:
        if rotate:
            img = ndimage.rotate(img, 90, reshape=False)
            img = resize(img, (img_size[0], img_size[1], img_size[2]), order=0)
            img = np.array(img)
        else:
            img = resize(img, (img_size[0], img_size[1], img_size[2]), order=0)
            img = np.array(img)
    if dim == 4:
        img = resize(img, (img_size[0], img_size[1], img_size[2], img.shape[3]),
                     order=0)
        img = np.array(img)
    if np.min(img) < np.max(img):
        img = img - np.min(img)
        img = img / np.max(img)

    return img


def read_nifti_seg_file(filepath):
    # Read file
    img = nib.load(filepath)
    # Get raw data H W D C
    img = img.get_fdata().astype(np.float32)
    dim = len(img.shape)
    if dim == 3:
        img = np.expand_dims(img, axis=0)
    if dim == 4:
        img = np.array(img)
        # H W D C -> C H W D
        img = img.transpose(3, 0, 1, 2)

    mask = img.sum(0) > 0
    for k in range(img.shape[0]):
        x = img[k, ...]
        y = x[mask]

        # 对背景外的区域进行归一化
        x[mask] -= y.mean()
        x[mask] /= y.std()
        img[k, ...] = x

    return img


def get_cls_label(file_path):
    label_dir, file_name = os.path.split(file_path)
    hospital_dir, label = os.path.split(label_dir)

    return label


def get_nifti_seg_label(file_path):
    label = nib.load(file_path)
    label = label.get_fdata().astype(np.int64)

    return label


def read_seg_img_label(x):
    sample = {
        'image': read_nifti_seg_file(x[0]),
        'label': get_nifti_seg_label(x[1])
    }
    transform = transforms.Compose([
        RandomRotFlip(),
        RandomCrop((img_size[0], img_size[1], img_size[2])),
        GaussianNoise(p=0.1),
        ToTensor(re.sub('\D', '', x[0]))
    ])
    transform(sample)
    return 1


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.stack([np.rot90(x, k) for x in image], axis=0)
        label = np.rot90(label, k)
        axis = np.random.randint(1, 4)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis - 1).copy()

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):

        image, label = sample['image'], sample['label']
        (c, h, w, d) = image.shape
        print(h, w, d)
        print(self.output_size)
        h1 = np.random.randint(0, h - self.output_size[0])
        w1 = np.random.randint(0, w - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[h1:h1 + self.output_size[0], w1:w1 + self.output_size[1],
                      d1:d1 + self.output_size[2]]
        image = image[:, h1:h1 + self.output_size[0],
                      w1:w1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + np.random.normal(
        0.0, variance, size=data_sample.shape)
    return data_sample


class GaussianNoise(object):

    def __init__(self, noise_variance=(0, 0.1), p=0.5):
        self.prob = p
        self.noise_variance = noise_variance

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if np.random.uniform() < self.prob:
            image = augment_gaussian_noise(image, self.noise_variance)
        return {'image': image, 'label': label}


#######################################----------End utils functions---------------##########################

#######################################------Start Dataset class------###########################


class CTSegDataset(Dataset):

    def __init__(self, imgpath_path, label_path=None):
        print(
            "<<<<<<<<<<<<<<<<<<<<Waiting For Images Loading<<<<<<<<<<<<<<<<<<<<"
            + "\n")
        img_names = [
            os.path.join(imgpath_path, img_name)
            for img_name in tqdm(os.listdir(imgpath_path))
            if img_name.startswith('.') == False
        ]
        label_names = [
            os.path.join(label_path, label_name)
            for label_name in tqdm(os.listdir(label_path))
            if label_name.startswith('.') == False
        ]
        ziped_results = list(zip(img_names, label_names))

        ziped_results = list(zip(img_names, label_names))
        pool = Pool(processes=number_threads)
        result = []
        for res in ziped_results:
            result.append(pool.apply_async(read_seg_img_label, (res,)))
        pool.close()
        pool.join()
        result = [res.get() for res in result]


class CTClsDataset(Dataset):

    def __init__(self, srcpath):
        self.srcpath = srcpath
        self.dataset = {"image": [], "label": []}
        self.transform = transforms.ToTensor()
        print(
            "<<<<<<<<<<<<<<<<<<<<Waiting For Data Loading<<<<<<<<<<<<<<<<<<<<" +
            "\n")
        for roots, dirs, files in os.walk(self.srcpath):
            for file in tqdm(files):
                file_path = os.path.join(roots, file)
                label = int(get_cls_label(file_path))
                image = read_nifti_cls_file(file_path, rotate)

                if len(image.shape) == 3:
                    # 增加一维channel，确保tensor输入为NCDHW
                    image = np.expand_dims(image, axis=0)
                    image = torch.from_numpy(image)
                    image = torch.permute(image, (0, 3, 1, 2))
                else:
                    image = image.astype(np.float32)
                    image = torch.from_numpy(image)
                    image = torch.permute(image, (3, 2, 0, 1))

                self.dataset['image'].append(image)
                self.dataset['label'].append(label)

        self.dataset["label"] = torch.LongTensor(self.dataset['label'])
        print("<<<<<<<<<<<<<<<<<<<<Data Loading Complete<<<<<<<<<<<<<<<<<<<<" +
              "\n")

    def __getitem__(self, index):
        image, label = self.dataset['image'][index], self.dataset['label'][
            index]

        return image, label

    def __len__(self):
        return len(self.dataset['image'])


#######################################------End Dataset class------###########################


def load_data_from_path(imgpath_path, label_path, task, data_type):

    if task == "classification" and data_type == "CT":
        train_dataset = CTClsDataset(imgpath_path)

        return trainloader

    if task == "classification" and data_type == "RGB":
        train_dataset = RGBClsImage(train_path)
        test_dataset = RGBClsImage(test_path)
        trainloader, testloader, num_examples = data_loader(
            train_dataset, test_dataset)

        return trainloader, testloader, num_examples

    if task == "segmentation" and data_type == "CT":
        train_dataset = CTSegDataset(imgpath_path, label_path)
        return train_dataset


if __name__ == '__main__':
    data = load_data_from_path(imgpath_path, label_path, task, data_type)
