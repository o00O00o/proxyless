import os
import json
import torch
import argparse
import numpy as np
from run_manager import RunConfig, RunManager
from model.proxyless_nets import ProxylessNASNets


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/home/gaoyibo/codes/proxyless/record/search_baseline/')
parser.add_argument('--n_epochs', type=int, default=600)
parser.add_argument('--train_from_scratch', action='store_true')
parser.add_argument('--manual_seed', default=0, type=int)
parser.add_argument('--resume', action='store_true')


def run_exp(args, exp_path):
    # prepare run config
    run_config_path = os.path.join(exp_path, 'run.config')
    run_config = json.load(open(run_config_path, 'r'))
    run_config['n_epochs'] = args.n_epochs
    run_config['dataset'] = 'cifar10'
    run_config = RunConfig(**run_config)

    # prepare network
    net_config_path = os.path.join(exp_path, 'net.config')
    net_config = json.load(open(net_config_path, 'r'))
    net = ProxylessNASNets.build_from_config(net_config)

    # build run manager
    run_manager = RunManager(exp_path, net, run_config)

    # load checkpoints
    init_path = os.path.join(exp_path, 'init')
    if args.resume:
        run_manager.load_model()
    elif os.path.isfile(init_path) and not args.train_from_scratch:
        search_weight = torch.load(init_path)['state_dict']
        train_weight = run_manager.net.module.state_dict()
        for key in list(search_weight.keys()):
            if not key.startswith('classifier'):
                train_weight[key] = search_weight[key]
    else:
        print('Random initialization, train from scratch.')

    # train
    print('Start training')
    run_manager.train(print_top5=True)
    run_manager.save_model()

    # test
    print('Test on test set')
    loss, acc1, acc5 = run_manager.validate(is_test=True, return_top5=True)
    log = 'test_loss: %f\t test_acc1: %f\t test_acc5: %f' % (loss, acc1, acc5)
    run_manager.write_log(log, prefix='test')
    output_dict = {
        'test_loss': '%f' % loss, 'test_acc1': '%f' % acc1, 'test_acc5': '%f' % acc5
    }
    json.dump(output_dict, open(os.path.join(exp_path, 'output'), 'w'), indent=4)


if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)

    assert os.path.exists(args.path), print('Exp record not found.')

    exp_path_list = []
    for dir_name in os.listdir(args.path):
        if dir_name.startswith('learned_net') and int(dir_name.split('_')[-1]) > 200:
            exp_path_list.append(os.path.join(args.path, dir_name))
    
    exp_path_list = sorted(exp_path_list)
    for exp_path in exp_path_list:
        run_exp(args, exp_path)
