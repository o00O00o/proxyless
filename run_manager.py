# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

from model.proxyless_nets import ProxylessNASNets
import os
import time
import json
import torch.optim
import numpy as np
from utils import *
from datetime import timedelta
import torch.backends.cudnn as cudnn


class RunConfig:

    def __init__(self, data_path, dataset, n_epochs, init_lr, lr_schedule_type, batch_size, 
                 train_ratio, n_worker, label_smoothing, model_init, validation_frequency, 
                 print_frequency, **kwargs):
        
        self.data_path = data_path
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.n_worker = n_worker

        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.lr_schedule_type = lr_schedule_type
        self.label_smoothing = label_smoothing
        self.model_init = model_init

        self.validation_frequency = validation_frequency
        self.print_frequency = print_frequency

        self._data_provider = None
        self._train_iter, self._valid_iter, self._test_iter = None, None, None

    @property
    def config(self):
        config = {}
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    """ learning rate """

    def _calc_learning_rate(self, epoch, n_epochs, batch=0, nBatch=None):
        if self.lr_schedule_type == 'cosine':
            T_total = n_epochs * nBatch
            T_cur = epoch * nBatch + batch
            lr = 0.5 * self.init_lr * (1 + math.cos(math.pi * T_cur / T_total))
        else:
            raise ValueError('do not support: %s' % self.lr_schedule_type)
        return lr

    def adjust_learning_rate(self, optimizer, epoch, n_epochs, batch=0, nBatch=None):
        """ adjust learning of a given optimizer and return the new learning rate """
        new_lr = self._calc_learning_rate(epoch, n_epochs, batch, nBatch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    """ data provider """

    @property
    def data_config(self):
        return {
            'data_path': self.data_path,
            'batch_size': self.batch_size,
            'train_ratio': self.train_ratio,
            'n_worker': self.n_worker,
        }

    @property
    def data_provider(self):
        if self._data_provider is None:
            if self.dataset == 'cifar10':
                from data_providers import Cifar10DataProvider
                self._data_provider = Cifar10DataProvider(**self.data_config)
            elif self.dataset == 'SimCLR':
                from data_providers import SimCLRDataProvider
                self._data_provider = SimCLRDataProvider(**self.data_config)
            else:
                raise ValueError('do not support: %s' % self.dataset)
        return self._data_provider

    def change_dataset_batchsize(self, dataset, batch_size):
        self._data_provider = None
        self.dataset = dataset
        self.batch_size = batch_size

    @property
    def train_loader(self):
        return self.data_provider.train

    @property
    def valid_loader(self):
        return self.data_provider.valid

    @property
    def test_loader(self):
        return self.data_provider.test

    @property
    def train_next_batch(self):
        if self._train_iter is None:
            self._train_iter = iter(self.train_loader)
        try:
            data = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            data = next(self._train_iter)
        return data

    @property
    def valid_next_batch(self):
        if self._valid_iter is None:
            self._valid_iter = iter(self.valid_loader)
        try:
            data = next(self._valid_iter)
        except StopIteration:
            self._valid_iter = iter(self.valid_loader)
            data = next(self._valid_iter)
        return data

    @property
    def test_next_batch(self):
        if self._test_iter is None:
            self._test_iter = iter(self.test_loader)
        try:
            data = next(self._test_iter)
        except StopIteration:
            self._test_iter = iter(self.test_loader)
            data = next(self._test_iter)
        return data

    """ optimizer """

    def build_optimizer(self, net_params):
        optimizer = torch.optim.Adam(net_params, lr=self.init_lr)
        return optimizer


class RunManager:

    def __init__(self, path, net, run_config: RunConfig, out_log=True):
        self.path = path
        self.net = net
        self.run_config = run_config
        self.out_log = out_log

        self._logs_path, self._save_path = None, None
        self.best_acc = 0
        self.start_epoch = 0

        # initialize model (default)
        self.net.init_model(run_config.model_init)

        # move network to GPU if available
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            self.net.to(self.device)
            cudnn.benchmark = True
        else:
            raise ValueError

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.run_config.build_optimizer(self.net.weight_parameters())

    """ save path and log path """

    @property
    def save_path(self):
        if self._save_path is None:
            save_path = os.path.join(self.path, 'checkpoint')
            os.makedirs(save_path, exist_ok=True)
            self._save_path = save_path
        return self._save_path

    @property
    def logs_path(self):
        if self._logs_path is None:
            logs_path = os.path.join(self.path, 'logs')
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return self._logs_path

    """ save and load models """

    def save_model(self, checkpoint=None, is_best=False, model_name=None):
        if checkpoint is None:
            checkpoint = {'state_dict': self.net.state_dict()}

        if model_name is None:
            model_name = 'checkpoint.pth.tar'

        checkpoint['dataset'] = self.run_config.dataset  # add `dataset` info to the checkpoint
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        model_path = os.path.join(self.save_path, model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        torch.save(checkpoint, model_path)

        if is_best:
            best_path = os.path.join(self.save_path, 'model_best.pth.tar')
            torch.save({'state_dict': checkpoint['state_dict']}, best_path)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]

        try:
            if model_fname is None or not os.path.exists(model_fname):
                model_fname = '%s/checkpoint.pth.tar' % self.save_path
                with open(latest_fname, 'w') as fout:
                    fout.write(model_fname + '\n')
            if self.out_log:
                print("=> loading checkpoint '{}'".format(model_fname))

            if torch.cuda.is_available():
                checkpoint = torch.load(model_fname)
            else:
                checkpoint = torch.load(model_fname, map_location='cpu')

            self.net.load_state_dict(checkpoint['state_dict'])
            # set new manual seed
            new_manual_seed = int(time.time())
            torch.manual_seed(new_manual_seed)
            torch.cuda.manual_seed_all(new_manual_seed)
            np.random.seed(new_manual_seed)

            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
            if 'best_acc' in checkpoint:
                self.best_acc = checkpoint['best_acc']
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            if self.out_log:
                print("=> loaded checkpoint '{}'".format(model_fname))
        except Exception:
            if self.out_log:
                print('fail to load checkpoint from %s' % self.save_path)

    def save_config(self, print_info=True):
        """ dump run_config and net_config to the model_folder """
        os.makedirs(self.path, exist_ok=True)
        net_save_path = os.path.join(self.path, 'net.config')
        json.dump(self.net.config, open(net_save_path, 'w'), indent=4)
        if print_info:
            print('Network configs dump to %s' % net_save_path)

        run_save_path = os.path.join(self.path, 'run.config')
        json.dump(self.run_config.config, open(run_save_path, 'w'), indent=4)
        if print_info:
            print('Run configs dump to %s' % run_save_path)

    """ train and test """

    def write_log(self, log_str, prefix, should_print=True):
        """ prefix: valid, train, test """
        if prefix in ['valid', 'test']:
            with open(os.path.join(self.logs_path, 'valid_console.txt'), 'a') as fout:
                fout.write(log_str + '\n')
                fout.flush()
        if prefix in ['valid', 'test', 'train']:
            with open(os.path.join(self.logs_path, 'train_console.txt'), 'a') as fout:
                if prefix in ['valid', 'test']:
                    fout.write('=' * 10)
                fout.write(log_str + '\n')
                fout.flush()
        if should_print:
            print(log_str)

    def validate(self, is_test=True, use_train_mode=False, return_top5=False):
        data_loader = self.run_config.valid_loader

        if use_train_mode:
            self.net.train()
        else:
            self.net.eval()
        
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                # compute output
                output = self.net(images)
                loss = self.criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.run_config.print_frequency == 0 or i + 1 == len(data_loader):
                    if is_test:
                        prefix = 'Test'
                    else:
                        prefix = 'Valid'
                    test_log = prefix + ': [{0}/{1}]\t'\
                                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'\
                                        'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})'.\
                        format(i, len(data_loader) - 1, batch_time=batch_time, loss=losses, top1=top1)
                    if return_top5:
                        test_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top5)
                    print(test_log)
        if return_top5:
            return losses.avg, top1.avg, top5.avg
        else:
            return losses.avg, top1.avg

    def train_one_epoch(self, adjust_lr_func, train_log_func):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        self.net.train()

        end = time.time()
        for i, (images, labels) in enumerate(self.run_config.train_loader):
            data_time.update(time.time() - end)
            new_lr = adjust_lr_func(i)
            images, labels = images.to(self.device), labels.to(self.device)

            # compute output
            output = self.net(images)
            if self.run_config.label_smoothing > 0:
                loss = cross_entropy_with_label_smoothing(output, labels, self.run_config.label_smoothing)
            else:
                loss = self.criterion(output, labels)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss, images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            self.net.zero_grad()  # or self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.run_config.print_frequency == 0 or i + 1 == len(self.run_config.train_loader):
                batch_log = train_log_func(i, batch_time, data_time, losses, top1, top5, new_lr)
                self.write_log(batch_log, 'train')
        return top1, top5

    def train(self, print_top5=False):
        nBatch = len(self.run_config.train_loader)

        def train_log_func(epoch_, i, batch_time, data_time, losses, top1, top5, lr):
            batch_log = 'Train [{0}][{1}/{2}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                        'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                        'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})'. \
                format(epoch_ + 1, i, nBatch - 1,
                       batch_time=batch_time, data_time=data_time, losses=losses, top1=top1)
            if print_top5:
                batch_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top5)
            batch_log += '\tlr {lr:.5f}'.format(lr=lr)
            return batch_log

        for epoch in range(self.start_epoch, self.run_config.n_epochs):
            print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')

            end = time.time()
            train_top1, train_top5 = self.train_one_epoch(
                lambda i: self.run_config.adjust_learning_rate(self.optimizer, epoch, self.run_config.n_epochs, i, nBatch),
                lambda i, batch_time, data_time, losses, top1, top5, new_lr:
                train_log_func(epoch, i, batch_time, data_time, losses, top1, top5, new_lr),
            )
            time_per_epoch = time.time() - end
            seconds_left = int((self.run_config.n_epochs - epoch - 1) * time_per_epoch)
            print('Time per epoch: %s, Est. complete in: %s' % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))

            if (epoch + 1) % self.run_config.validation_frequency == 0:
                val_loss, val_acc, val_acc5 = self.validate(is_test=False, return_top5=True)
                is_best = val_acc > self.best_acc
                self.best_acc = max(self.best_acc, val_acc)
                val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})'.\
                    format(epoch + 1, self.run_config.n_epochs, val_loss, val_acc, self.best_acc)
                if print_top5:
                    val_log += '\ttop-5 acc {0:.3f}\tTrain top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}'.\
                        format(val_acc5, top1=train_top1, top5=train_top5)
                else:
                    val_log += '\tTrain top-1 {top1.avg:.3f}'.format(top1=train_top1)
                self.write_log(val_log, 'valid')
            else:
                is_best = False

            self.save_model({
                'epoch': epoch,
                'best_acc': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
                'state_dict': self.net.state_dict(),
            }, is_best=is_best)


    def weight_stablize(self, mode, stablize_epochs, end_epochs=800):
        init_path = os.path.join(self.path, 'init')
        checkpoint = torch.load(init_path)
        self.net.load_state_dict(checkpoint['state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        nBatch = len(self.run_config.train_loader)

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        self.net.train()
        epoch_count = 0

        for epoch in range(start_epoch, start_epoch + stablize_epochs):
            epoch_count += 1
            for i, (images, labels) in enumerate(self.run_config.train_loader):
                new_lr = self.run_config.adjust_learning_rate(self.optimizer, epoch, end_epochs, i, nBatch)

                if mode == 'supervised':
                    images, labels = images.to(self.device), labels.to(self.device)
                    logits = self.net(images)
                    if self.run_config.label_smoothing > 0:
                        loss = cross_entropy_with_label_smoothing(logits, labels, self.run_config.label_smoothing)
                    else:
                        loss = self.criterion(logits, labels)
                elif mode == 'SimCLR':
                    images = torch.cat(images, dim=0)
                    images = images.to(self.device)
                    features = self.net(images)
                    logits, labels = info_nce_loss(features, self.device)
                    loss = self.criterion(logits, labels)
                else:
                    raise NotImplementedError

                # measure accuracy and record loss
                acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
                losses.update(loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # compute gradient and do SGD step
                self.net.zero_grad()  # or self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                 
                if i % self.run_config.print_frequency == 0 or i + 1 == len(self.run_config.train_loader):
                    batch_log = 'Stablize [{0}/{1}][{2}/{3}]\t' \
                                'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                                'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})\t' \
                                'Top-5 acc {top5.val:.3f} ({top5.avg:.3f})\tlr {lr:.5f}'. \
                                format(epoch_count, stablize_epochs, i, nBatch - 1, losses=losses, top1=top1, top5=top5, lr=new_lr)
                    print(batch_log)
        
