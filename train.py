import os
import json
import torch
import argparse
import numpy as np
from shutil import copytree
from run_manager import RunConfig, RunManager
from model.proxyless_nets import ProxylessNASNets


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/home/gaoyibo/codes/proxyless/record/search_baseline/')
parser.add_argument('--weight_preserve', action='store_true')
parser.add_argument('--manual_seed', default=0, type=int)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--stablize_epochs', type=int, default=1)
parser.add_argument('--mode', type=str, default='SimCLR', choices=['supervised', 'SimCLR'])

""" run config """
parser.add_argument('--data_path', type=str, default='/home/gaoyibo/Datasets/cifar-10/')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'SimCLR'])
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--train_ratio', type=int, default=0.8)
parser.add_argument('--n_worker', type=int, default=16)

parser.add_argument('--n_epochs', type=int, default=600)
parser.add_argument('--init_lr', type=float, default=1e-3)
parser.add_argument('--lr_schedule_type', type=str, default='cosine')
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--model_init', type=str, default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--validation_frequency', type=int, default=1)
parser.add_argument('--print_frequency', type=int, default=100)


def run_exp(args, exp_path):

    # generate a new dir for training
    if args.weight_preserve:
        weight_status = 'weight_preserve'
    else:
        weight_status = 'train_from_scratch'
    exp_name = args.mode + '_' + weight_status
    parent_path = os.path.realpath(os.path.join(args.path, os.pardir))
    new_path = os.path.join(parent_path, exp_name)
    copytree(args.path, new_path)
    args.path = new_path

    # prepare run config
    run_config = RunConfig(**args.__dict__)

    # prepare network
    net_config_path = os.path.join(exp_path, 'net.config')

    if args.resume:
        net = ProxylessNASNets.build_from_config(net_config_path, is_supervised=True)
        run_manager = RunManager(exp_path, net, run_config)
        run_manager.load_model()
    elif args.weight_preserve:
        print('Stablize the preserved weights')
        if args.mode == 'SimCLR':
            net = ProxylessNASNets.build_from_config(net_config_path, is_supervised=False)
            run_config.dataset = 'SimCLR'
            run_manager = RunManager(exp_path, net, run_config)
            run_manager.weight_stablize(args.mode, args.stablize_epochs)
            # load the stablized weights to supervised model
            search_weight = run_manager.net.state_dict()
            run_manager.net = ProxylessNASNets.build_from_config(net_config_path, is_supervised=True).to(run_manager.device)
            train_weight = run_manager.net.state_dict()
            for key in list(search_weight.keys()):
                if not key.startswith('classifier'):
                    train_weight[key] = search_weight[key]
            # change the dataset to supervised dataset
            run_manager.run_config.change_dataset('cifar10')
        elif args.mode == 'supervised':
            net = ProxylessNASNets.build_from_config(net_config_path, is_supervised=True)
            run_manager = RunManager(exp_path, net, run_config)
            run_manager.weight_stablize(args.mode, args.stablize_epochs)
    else:
        print('Random initialization, train from scratch.')
        net = ProxylessNASNets.build_from_config(net_config_path, is_supervised=True)
        run_manager = RunManager(exp_path, net, run_config)

    # train
    print('\nStart training')
    run_manager.train(print_top5=True)
    run_manager.save_model()

    # test
    print('Test on test set')
    loss, acc1, acc5 = run_manager.validate(is_test=True, return_top5=True)
    log = 'test_loss: %f\t test_acc1: %f\t test_acc5: %f' % (loss, acc1, acc5)
    run_manager.write_log(log, prefix='test')
    output_dict = {'test_loss': '%f' % loss, 'test_acc1': '%f' % acc1, 'test_acc5': '%f' % acc5}
    json.dump(output_dict, open(os.path.join(exp_path, 'output'), 'w'), indent=4)


if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)

    assert os.path.exists(args.path), print('Exp record not found.')

    exp_path_list = []
    for dir_name in os.listdir(args.path):
        if dir_name.startswith('learned_net'):
            exp_path_list.append(os.path.join(args.path, dir_name))
    
    exp_path_list = sorted(exp_path_list, key=lambda path: int(path.split('_')[-1]))
    for exp_path in exp_path_list:
        run_exp(args, exp_path)
