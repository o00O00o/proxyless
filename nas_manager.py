import os
import time
import json
import torch
import numpy as np
from utils import *
from model.mix_op import MixedEdge
from run_manager import RunConfig, RunManager


class ArchSearchConfig:

    def __init__(self, arch_init_type, arch_init_ratio, arch_lr, grad_update_arch_param_every=1,
                 grad_update_steps=1, grad_binary_mode='full', **kwargs):
        """ architecture parameters initialization & optimizer """
        self.arch_init_type = arch_init_type
        self.arch_init_ratio = arch_init_ratio

        self.lr = arch_lr
        self.update_arch_param_every = grad_update_arch_param_every
        self.update_steps = grad_update_steps
        self.binary_mode = grad_binary_mode

    def build_optimizer(self, params):
        return torch.optim.Adam(params, self.lr)
    
    def get_update_schedule(self, nBatch):
        schedule = {}
        for i in range(nBatch):
            if (i + 1) % self.update_arch_param_every == 0:
                schedule[i] = self.update_steps
        return schedule


class ArchSearchRunManager:

    def __init__(self, mode, path, super_net, run_config: RunConfig, arch_search_config: ArchSearchConfig):
        # init weight parameters & build weight_optimizer
        self.mode = mode
        self.run_manager = RunManager(path, super_net, run_config, True)
        self.arch_search_config = arch_search_config

        # init architecture parameters
        self.net.init_arch_params(self.arch_search_config.arch_init_type, self.arch_search_config.arch_init_ratio,)

        # build architecture optimizer
        self.arch_optimizer = self.arch_search_config.build_optimizer(self.net.architecture_parameters())

        self.warmup = True
        self.warmup_epoch = 0

        self.highest_acc = 0
        self.lowest_loss = np.inf

    @property
    def net(self):
        return self.run_manager.net

    def write_log(self, log_str, prefix, should_print=True, end='\n'):
        with open(os.path.join(self.run_manager.logs_path, '%s.log' % prefix), 'a') as fout:
            fout.write(log_str + end)
            fout.flush()
        if should_print:
            print(log_str)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.run_manager.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]

        if model_fname is None or not os.path.exists(model_fname):
            model_fname = '%s/checkpoint.pth.tar' % self.run_manager.save_path
            with open(latest_fname, 'w') as fout:
                fout.write(model_fname + '\n')
        if self.run_manager.out_log:
            print("=> loading checkpoint '{}'".format(model_fname))

        if torch.cuda.is_available():
            checkpoint = torch.load(model_fname)
        else:
            checkpoint = torch.load(model_fname, map_location='cpu')

        model_dict = self.net.state_dict()
        model_dict.update(checkpoint['state_dict'])
        self.net.load_state_dict(model_dict)
        if self.run_manager.out_log:
            print("=> loaded checkpoint '{}'".format(model_fname))

        # set new manual seed
        new_manual_seed = int(time.time())
        torch.manual_seed(new_manual_seed)
        torch.cuda.manual_seed_all(new_manual_seed)
        np.random.seed(new_manual_seed)

        if 'epoch' in checkpoint:
            self.run_manager.start_epoch = checkpoint['epoch'] + 1
        if 'weight_optimizer' in checkpoint:
            self.run_manager.optimizer.load_state_dict(checkpoint['weight_optimizer'])
        if 'arch_optimizer' in checkpoint:
            self.arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])
        if 'warmup' in checkpoint:
            self.warmup = checkpoint['warmup']
        if self.warmup and 'warmup_epoch' in checkpoint:
            self.warmup_epoch = checkpoint['warmup_epoch']
    
    def save_arch_weights(self, epoch, note):
        # convert to normal network according to architecture parameters
        new_net = self.net.get_clone_net()
        normal_net = new_net.convert_to_normal_net(self.net)
        print('Total training params: %.2fM' % (count_parameters(normal_net) / 1e6))

        if note == 'epoch_save':
            learned_net_path = os.path.join(self.run_manager.path, 'learned_net_{}'.format(epoch + 1))
        elif note == 'highest_acc' or note == 'lowest_loss':
            learned_net_path = os.path.join(self.run_manager.path, note)
        else:
            raise NotImplementedError

        os.makedirs(learned_net_path, exist_ok=True)
        json.dump(normal_net.config, open(os.path.join(learned_net_path, 'net.config'), 'w'), indent=4)
        torch.save({'state_dict': normal_net.state_dict(), 'epoch': epoch}, os.path.join(learned_net_path, 'init'))


    """ training related methods """

    def validate(self):
        # get performances of current chosen network on validation set
        data_loader = self.run_manager.run_config.valid_loader
        data_loader.batch_sampler.drop_last = False
        # set chosen op active
        self.net.set_chosen_op_active()
        # remove unused modules
        self.net.unused_modules_off()
        # test on validation set under train mode
        self.net.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):

                if self.mode == 'supervised':
                    images, labels = images.to(self.run_manager.device), labels.to(self.run_manager.device)
                    logits = self.run_manager.net(images)
                    if self.run_manager.run_config.label_smoothing > 0:
                        loss = cross_entropy_with_label_smoothing(logits, labels, self.run_manager.run_config.label_smoothing)
                    else:
                        loss = self.run_manager.criterion(logits, labels)
                elif self.mode == 'SimCLR':
                    images = torch.cat(images, dim=0)
                    images = images.to(self.run_manager.device)
                    features = self.run_manager.net(images)
                    logits, labels = info_nce_loss(features, self.run_manager.device)
                    loss = self.run_manager.criterion(logits, labels)
                else:
                    raise NotImplementedError

                # measure accuracy and record loss
                acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
                losses.update(loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.run_manager.run_config.print_frequency == 0 or i + 1 == len(data_loader):
                    test_log = 'Valid: [{0}/{1}]\t' \
                               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'\
                               'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})'.\
                        format(i, len(data_loader) - 1, batch_time=batch_time, loss=losses, top1=top1)
                    test_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top5)
                    print(test_log)
        
        # unused modules back
        self.net.unused_modules_back()
        return losses.avg, top1.avg, top5.avg

    def warm_up(self, warmup_epochs=25):
        lr_max = 0.05
        data_loader = self.run_manager.run_config.train_loader
        nBatch = len(data_loader)
        T_total = warmup_epochs * nBatch

        for epoch in range(self.warmup_epoch, warmup_epochs):
            print('\n', '-' * 30, 'Warmup epoch: %d' % (epoch + 1), '-' * 30, '\n')
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            # switch to train mode
            self.run_manager.net.train()

            end = time.time()
            for i, (images, labels) in enumerate(data_loader):
                data_time.update(time.time() - end)
                # lr
                T_cur = epoch * nBatch + i
                warmup_lr = 0.5 * lr_max * (1 + math.cos(math.pi * T_cur / T_total))
                for param_group in self.run_manager.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                
                self.net.reset_binary_gates()  # random sample binary gates
                self.net.unused_modules_off()  # remove unused module for speedup

                if self.mode == 'supervised':
                    images, labels = images.to(self.run_manager.device), labels.to(self.run_manager.device)
                    logits = self.run_manager.net(images)  # forward (DataParallel)
                    if self.run_manager.run_config.label_smoothing > 0:
                        loss = cross_entropy_with_label_smoothing(logits, labels, self.run_manager.run_config.label_smoothing)
                    else:
                        loss = self.run_manager.criterion(logits, labels)
                elif self.mode == 'SimCLR':
                    images = torch.cat(images, dim=0)
                    images = images.to(self.run_manager.device)
                    features = self.run_manager.net(images)  # forward (DataParallel)
                    logits, labels = info_nce_loss(features, self.run_manager.device)
                    loss = self.run_manager.criterion(logits, labels)
                else:
                    raise NotImplementedError
                
                # measure accuracy and record loss
                acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
                losses.update(loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                # compute gradient and do SGD step
                self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
                loss.backward()
                self.run_manager.optimizer.step()  # update weight parameters
                # unused modules back
                self.net.unused_modules_back()
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.run_manager.run_config.print_frequency == 0 or i + 1 == nBatch:
                    batch_log = 'Warmup Train [{0}][{1}/{2}]\t' \
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                                'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                                'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})\t' \
                                'Top-5 acc {top5.val:.3f} ({top5.avg:.3f})\tlr {lr:.5f}'. \
                        format(epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time,
                               losses=losses, top1=top1, top5=top5, lr=warmup_lr)
                    self.run_manager.write_log(batch_log, 'train')
            valid_res = self.validate()
            val_log = 'Warmup Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f}\ttop-5 acc {4:.3f}\t' \
                      'Train top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}'. \
                format(epoch + 1, warmup_epochs, *valid_res, top1=top1, top5=top5)
            self.run_manager.write_log(val_log, 'valid')
            self.warmup = epoch + 1 < warmup_epochs

            state_dict = self.net.state_dict()
            # rm architecture parameters & binary gates
            for key in list(state_dict.keys()):
                if 'AP_path_alpha' in key or 'AP_path_wb' in key:
                    state_dict.pop(key)
            checkpoint = {'state_dict': state_dict, 'warmup': self.warmup}
            if self.warmup:
                checkpoint['warmup_epoch'] = epoch,
            self.run_manager.save_model(checkpoint, model_name='warmup.pth.tar')

    def train(self, save_times):
        data_loader = self.run_manager.run_config.train_loader
        nBatch = len(data_loader)

        arch_param_num = len(list(self.net.architecture_parameters()))
        binary_gates_num = len(list(self.net.binary_gates()))
        weight_param_num = len(list(self.net.weight_parameters()))
        print(
            '#arch_params: %d\t#binary_gates: %d\t#weight_params: %d' %
            (arch_param_num, binary_gates_num, weight_param_num)
        )

        update_schedule = self.arch_search_config.get_update_schedule(nBatch)
        n_epochs = self.run_manager.run_config.n_epochs

        if save_times is None:
            save_stamps = list(range(50, n_epochs, 50))
        else:
            save_stamps = np.linspace(n_epochs // 4, n_epochs, save_times).astype(int)

        for epoch in range(self.run_manager.start_epoch, self.run_manager.run_config.n_epochs):
            print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            entropy = AverageMeter()
            # switch to train mode
            self.run_manager.net.train()

            end = time.time()
            for i, (images, labels) in enumerate(data_loader):
                data_time.update(time.time() - end)
                # lr
                lr = self.run_manager.run_config.adjust_learning_rate(self.run_manager.optimizer, epoch, n_epochs, batch=i, nBatch=nBatch)
                # network entropy
                net_entropy = self.net.entropy()
                entropy.update(net_entropy.data.item() / arch_param_num, 1)

                self.net.reset_binary_gates()  # random sample binary gates
                self.net.unused_modules_off()  # remove unused module for speedup

                # train weight parameters
                if self.mode == 'supervised':
                    images, labels = images.to(self.run_manager.device), labels.to(self.run_manager.device)
                    logits = self.run_manager.net(images)
                    if self.run_manager.run_config.label_smoothing > 0:
                        loss = cross_entropy_with_label_smoothing(logits, labels, self.run_manager.run_config.label_smoothing)
                    else:
                        loss = self.run_manager.criterion(logits, labels)
                elif self.mode == 'SimCLR':
                    images = torch.cat(images, dim=0)
                    images = images.to(self.run_manager.device)
                    features = self.run_manager.net(images)
                    logits, labels = info_nce_loss(features, self.run_manager.device)
                    loss = self.run_manager.criterion(logits, labels)
                else:
                    raise NotImplementedError
                
                # measure accuracy and record loss
                acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
                losses.update(loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                # compute gradient and do SGD step
                self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
                loss.backward()
                self.run_manager.optimizer.step()  # update weight parameters
                # unused modules back
                self.net.unused_modules_back()
                # skip architecture parameter updates in the first epoch
                if epoch > 0:
                    # update architecture parameters according to update_schedule
                    for j in range(update_schedule.get(i, 0)):
                        start_time = time.time()
                        arch_loss = self.gradient_step()
                        used_time = time.time() - start_time
                        log_str = 'Architecture [%d-%d]\t Time %.4f\t Loss %.4f' % (epoch + 1, i, used_time, arch_loss)
                        self.write_log(log_str, prefix='gradient', should_print=False)
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                # training log
                if i % self.run_manager.run_config.print_frequency == 0 or i + 1 == nBatch:
                    batch_log = 'Train [{0}][{1}/{2}]\t' \
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                                'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                                'Entropy {entropy.val:.5f} ({entropy.avg:.5f})\t' \
                                'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})\t' \
                                'Top-5 acc {top5.val:.3f} ({top5.avg:.3f})\tlr {lr:.5f}'. \
                        format(epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time,
                               losses=losses, entropy=entropy, top1=top1, top5=top5, lr=lr)
                    self.run_manager.write_log(batch_log, 'train')

            # print current network architecture
            self.write_log('-' * 30 + 'Current Architecture [%d]' % (epoch + 1) + '-' * 30, prefix='arch')
            for idx, block in enumerate(self.net.blocks):
                self.write_log('%d. %s' % (idx, block.module_str), prefix='arch')
            self.write_log('-' * 60, prefix='arch')

            # validate
            if (epoch + 1) % self.run_manager.run_config.validation_frequency == 0:
                (val_loss, val_top1, val_top5) = self.validate()

                if val_top1 > self.highest_acc:
                    self.save_arch_weights(epoch, 'highest_acc')
                    self.highest_acc = val_top1
                
                if val_loss < self.lowest_loss:
                    self.save_arch_weights(epoch, 'lowest_loss')
                    self.lowest_loss = val_loss

                val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})\ttop-5 acc {5:.3f}\t' \
                          'Train top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}\t' \
                          'Entropy {entropy.val:.5f}\t'. \
                    format(epoch + 1, self.run_manager.run_config.n_epochs, val_loss, val_top1,
                           self.highest_acc, val_top5, entropy=entropy, top1=top1, top5=top5)
                self.run_manager.write_log(val_log, 'valid')

            # save model
            self.run_manager.save_model({
                'warmup': False,
                'epoch': epoch,
                'weight_optimizer': self.run_manager.optimizer.state_dict(),
                'arch_optimizer': self.arch_optimizer.state_dict(),
                'state_dict': self.net.state_dict()
            })

            if epoch + 1 in save_stamps:
                self.save_arch_weights(epoch, 'epoch_save')

    def gradient_step(self):
        self.run_manager.run_config.valid_loader.batch_sampler.drop_last = True
        # switch to train mode
        self.run_manager.net.train()
        # Mix edge mode
        MixedEdge.MODE = self.arch_search_config.binary_mode
        time1 = time.time()  # time

        self.net.reset_binary_gates()  # random sample binary gates
        self.net.unused_modules_off()  # remove unused module for speedup

        # sample a batch of data from validation set
        if self.mode == 'supervised':
            images, labels = self.run_manager.run_config.valid_next_batch
            images, labels = images.to(self.run_manager.device), labels.to(self.run_manager.device)
            time2 = time.time()
            output = self.run_manager.net(images)
            time3 = time.time()
            loss = self.run_manager.criterion(output, labels)
        elif self.mode == 'SimCLR':
            images, _ = self.run_manager.run_config.valid_next_batch
            images = torch.cat(images, dim=0)
            images = images.to(self.run_manager.device)
            time2 = time.time()
            features = self.run_manager.net(images)
            logits, labels = info_nce_loss(features, self.run_manager.device)
            time3 = time.time()
            loss = self.run_manager.criterion(logits, labels)

        # compute gradient and do SGD step
        self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
        loss.backward()
        # set architecture parameter gradients
        self.net.set_arch_param_grad()
        self.arch_optimizer.step()
        if MixedEdge.MODE == 'two':
            self.net.rescale_updated_arch_param()
        # back to normal mode
        self.net.unused_modules_back()
        MixedEdge.MODE = None
        time4 = time.time()  # time
        self.write_log('(%.4f, %.4f, %.4f)' % (time2 - time1, time3 - time2, time4 - time3), 'gradient', should_print=False, end='\t')
        return loss.data.item()
