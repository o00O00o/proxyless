import torch
import numpy as np
from utils import get_split_list
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from utils.data_aug import ContrastiveLearningViewGenerator, GaussianBlur


class DataProvider:
    VALID_SEED = 0  # random seed for the validation set

    @staticmethod
    def name():
        """ Return name of the dataset """
        raise NotImplementedError

    @property
    def data_shape(self):
        """ Return shape as python list of one data entry """
        raise NotImplementedError

    @property
    def n_classes(self):
        """ Return `int` of num classes """
        raise NotImplementedError

    @property
    def save_path(self):
        """ local path to save the data """
        raise NotImplementedError

    @property
    def data_url(self):
        """ link to download the data """
        raise NotImplementedError

    @staticmethod
    def random_sample_valid_set(train_labels, valid_size, n_classes):
        train_size = len(train_labels)
        assert train_size > valid_size

        g = torch.Generator()
        g.manual_seed(DataProvider.VALID_SEED)  # set random seed before sampling validation set
        rand_indexes = torch.randperm(train_size, generator=g).tolist()

        train_indexes, valid_indexes = [], []
        per_class_remain = get_split_list(valid_size, n_classes)

        for idx in rand_indexes:
            label = train_labels[idx]
            if isinstance(label, float):
                label = int(label)
            elif isinstance(label, np.ndarray):
                label = np.argmax(label)
            else:
                assert isinstance(label, int)
            if per_class_remain[label] > 0:
                valid_indexes.append(idx)
                per_class_remain[label] -= 1
            else:
                train_indexes.append(idx)
        return train_indexes, valid_indexes


class Cifar10DataProvider(DataProvider):
    def __init__(self, data_path, batch_size, train_ratio, n_worker):

        self._data_path = data_path

        train_dataset = CIFAR10(data_path, train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(self.image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
        ]))

        valid_dataset = CIFAR10(data_path, train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ]))

        test_dataset = CIFAR10(data_path, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ]))

        valid_size = int(len(train_dataset) - train_ratio * len(train_dataset))
        print(valid_size)
        
        train_indexes, valid_indexes = self.random_sample_valid_set(train_dataset.targets, valid_size, self.n_classes)
        train_sampler, valid_sampler = SubsetRandomSampler(train_indexes), SubsetRandomSampler(valid_indexes)

        self.train = DataLoader(train_dataset, batch_size, sampler=train_sampler, num_workers=n_worker)
        self.valid = DataLoader(valid_dataset, batch_size, sampler=valid_sampler, num_workers=n_worker)
        self.test = DataLoader(test_dataset, batch_size, num_workers=n_worker)

        print('train: ' + str(len(self.train.sampler)))
        print('valid: ' + str(len(self.valid.sampler)))
        print('test: ' + str(len(self.test.sampler)))
    
    @property
    def normalize(self):
        return transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])
    
    @property
    def image_size(self):
        return 32

    @property
    def n_classes(self):
        return 10
    
    @property
    def data_path(self):
        if self._data_path is None:
            self._data_path = '/home/gaoyibo/Datasets/cifar-10/'
        return self._data_path


class SimCLRDataProvider(DataProvider):
    def __init__(self, data_path, batch_size, train_ratio, n_worker, s=1):
        self._data_path = data_path

        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.image_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * self.image_size)),
                                              transforms.ToTensor()])

        train_dataset = CIFAR10(data_path, train=True, download=True, transform=ContrastiveLearningViewGenerator(data_transforms, n_views=2))

        valid_size = int(len(train_dataset) - train_ratio * len(train_dataset))
        train_indexes, valid_indexes = self.random_sample_valid_set(train_dataset.targets, valid_size, self.n_classes)
        train_sampler, valid_sampler = SubsetRandomSampler(train_indexes), SubsetRandomSampler(valid_indexes)

        self.train = DataLoader(train_dataset, batch_size, sampler=train_sampler, num_workers=n_worker)
        self.valid = DataLoader(train_dataset, batch_size, sampler=valid_sampler, num_workers=n_worker)

        print('train: ' + str(len(self.train.sampler)))
        print('valid: ' + str(len(self.valid.sampler)))
    
    @property
    def image_size(self):
        return 32

    @property
    def n_classes(self):
        return 10

    @property
    def data_path(self):
        if self._data_path is None:
            self._data_path = '/home/gaoyibo/Datasets/cifar-10/'
        return self._data_path


if __name__ == '__main__':
    provider = SimCLRDataProvider(64, 0.2, 16)
    for idx, (images, _) in enumerate(provider.train):
        images = torch.cat(images, dim=0)
        print(images.shape)
        break