import os
import torch
import argparse
import numpy as np
from run_manager import RunConfig
from model.super_proxyless import SuperProxylessNASNets
from nas_manager import ArchSearchConfig, ArchSearchRunManager


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/home/gaoyibo/codes/proxyless/record/default_exp/')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--debug', help='freeze the weight parameters', action='store_true')
parser.add_argument('--manual_seed', default=0, type=int)

""" run config """
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--save_times', type=int, default=4)
parser.add_argument('--init_lr', type=float, default=0.025)
parser.add_argument('--lr_schedule_type', type=str, default='cosine')

parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--valid_size', type=int, default=0.2)
parser.add_argument('--n_worker', type=int, default=16)

parser.add_argument('--opt_type', type=str, default='sgd', choices=['sgd'])
parser.add_argument('--momentum', type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov', action='store_true')  # opt_param
parser.add_argument('--weight_decay', type=float, default=4e-5)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--no_decay_keys', type=str, default=None, choices=[None, 'bn', 'bn#bias'])

parser.add_argument('--model_init', type=str, default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--validation_frequency', type=int, default=1)
parser.add_argument('--print_frequency', type=int, default=10)

""" net config """
parser.add_argument('--width_stages', type=str, default='24,40,80,96,192,320')
parser.add_argument('--n_cell_stages', type=str, default='4,4,4,4,4,1')
parser.add_argument('--stride_stages', type=str, default='2,2,2,1,2,1')
parser.add_argument('--width_mult', type=float, default=1.0)
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--bn_eps', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0)

# architecture search config
""" arch search algo and warmup """
parser.add_argument('--warmup_epochs', type=int, default=40)
parser.add_argument('--arch_init_type', type=str, default='normal', choices=['normal', 'uniform'])
parser.add_argument('--arch_init_ratio', type=float, default=1e-3)
parser.add_argument('--arch_opt_type', type=str, default='adam', choices=['adam'])
parser.add_argument('--arch_lr', type=float, default=1e-3)
parser.add_argument('--arch_adam_beta1', type=float, default=0)  # arch_opt_param
parser.add_argument('--arch_adam_beta2', type=float, default=0.999)  # arch_opt_param
parser.add_argument('--arch_adam_eps', type=float, default=1e-8)  # arch_opt_param
parser.add_argument('--arch_weight_decay', type=float, default=0)
""" Grad hyper-parameters """
parser.add_argument('--grad_update_arch_param_every', type=int, default=5)
parser.add_argument('--grad_update_steps', type=int, default=1)
parser.add_argument('--grad_binary_mode', type=str, default='full_v2', choices=['full_v2', 'full', 'two'])
parser.add_argument('--grad_data_batch', type=int, default=None)


if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)

    os.makedirs(args.path, exist_ok=True)

    # build run config from args
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    run_config = RunConfig(**args.__dict__)

    # debug, adjust run_config
    if args.debug:
        run_config.train_batch_size = 1024
        run_config.test_batch_size = 1024
        run_config.valid_size = 0.8
        run_config.n_worker = 10
        run_config.n_epochs = 12
        args.warmup_epochs = 0

    # build net from args
    args.width_stages = [int(val) for val in args.width_stages.split(',')]
    args.n_cell_stages = [int(val) for val in args.n_cell_stages.split(',')]
    args.stride_stages = [int(val) for val in args.stride_stages.split(',')]
    args.conv_candidates = [
        '3x3_MBConv3', '3x3_MBConv6',
        '5x5_MBConv3', '5x5_MBConv6',
        '7x7_MBConv3', '7x7_MBConv6',
    ]
    super_net = SuperProxylessNASNets(
        width_stages=args.width_stages, n_cell_stages=args.n_cell_stages, stride_stages=args.stride_stages,
        conv_candidates=args.conv_candidates, n_classes=run_config.data_provider.n_classes, width_mult=args.width_mult,
        bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout
    )

    # build arch search config from args
    if args.arch_opt_type == 'adam':
        args.arch_opt_param = {
            'betas': (args.arch_adam_beta1, args.arch_adam_beta2),
            'eps': args.arch_adam_eps,
        }
    else:
        args.arch_opt_param = None

    arch_search_config = ArchSearchConfig(**args.__dict__)

    # arch search run manager
    arch_search_run_manager = ArchSearchRunManager(args.path, super_net, run_config, arch_search_config)

    # resume
    if args.resume:
        arch_search_run_manager.load_model()

    # warmup
    if arch_search_run_manager.warmup:
        arch_search_run_manager.warm_up(warmup_epochs=args.warmup_epochs)

    # joint training
    arch_search_run_manager.train(args.save_times, fix_net_weights=False)
