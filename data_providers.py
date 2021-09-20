import numpy as np
from utils import *
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


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
    def __init__(self, save_path, train_batch_size, test_batch_size, valid_size, n_worker):

        train_dataset = CIFAR10(save_path, train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(self.image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
        ]))

        valid_dataset = CIFAR10(save_path, train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ]))

        test_dataset = CIFAR10(save_path, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ]))

        if isinstance(valid_size, float):
            valid_size = int(valid_size * len(train_dataset))
        else:
            assert isinstance(valid_size, int), 'invalid valid_size: %s' % valid_size
        
        train_indexes, valid_indexes = self.random_sample_valid_set(train_dataset.targets, valid_size, self.n_classes)
        train_sampler, valid_sampler = SubsetRandomSampler(train_indexes), SubsetRandomSampler(valid_indexes)

        self.train = DataLoader(train_dataset, train_batch_size, sampler=train_sampler, num_workers=n_worker)
        self.valid = DataLoader(valid_dataset, test_batch_size, sampler=valid_sampler, num_workers=n_worker)
        self.test = DataLoader(test_dataset, test_batch_size, num_workers=n_worker)

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
    def save_path(self):
        if self._save_path is None:
            self._save_path = '/home/gaoyibo/Datasets/cifar-10/'
        return self._save_path
