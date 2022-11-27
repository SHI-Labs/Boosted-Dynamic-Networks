import os
import glob
import time
import argparse
import numpy as np


model_names = ['msdnet', 'msdnet_ge', 'ranet']

arg_parser = argparse.ArgumentParser(description='Image classification PK main script')

exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--result_dir', default='results/exp_{}'.format(time.time()), type=str, metavar='SAVE', help='path to the experiment logging directory')
exp_group.add_argument('--tensorboard_dir', default='results/exp_{}'.format(time.time()), type=str, metavar='SAVE', help='path to the tensorboard logging directory')
exp_group.add_argument('--checkpoint', type=str, default='', help='initialize the model with a pretrained checkpoint.')
exp_group.add_argument('--resume', action='store_true', help='path to latest checkpoint (default: none)')
exp_group.add_argument('--evalmode', default=None, choices=['anytime', 'dynamic'], help='which mode to evaluate')
exp_group.add_argument('--evaluate-from', default=None, type=str, metavar='PATH', help='path to saved checkpoint (default: none)')
exp_group.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 100)')
exp_group.add_argument('--seed', default=0, type=int, help='random seed')
exp_group.add_argument('--gpu', default=None, type=str, help='GPU available.')

exp_group.add_argument('--flat_curve', action='store_true',
                       help='make curve flat if more flops does not get better performance under dyanmic evaluation.')
exp_group.add_argument('--save_suffix', default="", type=str, help='suffix when saving evaluation results.')

# dataset related
data_group = arg_parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--dataset', metavar='D', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'], help='data to work on')
data_group.add_argument('--data-root', metavar='DIR', default='data', help='path to dataset (default: data)')
data_group.add_argument('--use-valid', action='store_true', help='use validation set or not')
data_group.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
data_group.add_argument('--val_workers', default=1, type=int, metavar='N', help='number of data loading workers (default: 4)')

# model arch related
arch_group = arg_parser.add_argument_group('arch', 'model architecture setting')
arch_group.add_argument('--arch', metavar='ARCH', default='resnet', type=str, choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: msdnet)')
arch_group.add_argument('--reduction', default=0.5, type=float, metavar='C', help='compression ratio of DenseNet (1 means dot\'t use compression) (default: 0.5)')

# msdnet and some ranet config
arch_group.add_argument('--nBlocks', type=int, default=1)
arch_group.add_argument('--nChannels', type=int, default=32)
arch_group.add_argument('--base', type=int, default=4)
arch_group.add_argument('--stepmode', type=str, choices=['even', 'lin_grow'])
arch_group.add_argument('--step', type=int, default=1)
arch_group.add_argument('--growthRate', type=int, default=6)
arch_group.add_argument('--grFactor', default='1-2-4', type=str)
arch_group.add_argument('--prune', default='max', choices=['min', 'max'])
arch_group.add_argument('--bnFactor', default='1-2-4')
arch_group.add_argument('--bottleneck', default=True, type=bool)

# ranet config
arch_group.add_argument('--block-step', type=int, default=2)
arch_group.add_argument('--scale-list', default='1-2-3', type=str)
arch_group.add_argument('--compress-factor', default=0.25, type=float)
arch_group.add_argument('--bnAfter', action='store_true', default=True)

# training related
optim_group = arg_parser.add_argument_group('optimization','optimization setting')
optim_group.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run (default: 164)')
optim_group.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
optim_group.add_argument('--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 64)')
optim_group.add_argument('--optimizer', default='sgd', choices=['sgd', 'rmsprop', 'adam'], metavar='N', help='optimizer (default=sgd)')
optim_group.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate (default: 0.1)')
optim_group.add_argument('--lr-type', default='multistep', type=str, metavar='T', help='learning rate strategy (default: multistep)', choices=['cosine', 'multistep'])
optim_group.add_argument('--decay-rate', default=0.1, type=float, metavar='N', help='decay rate of learning rate (default: 0.1)')
optim_group.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (default=0.9)')
optim_group.add_argument('--weight-decay', '--wd', default='1e-4', type=str, metavar='W', help='weight decay (default: 1e-4)')

# boosting related
boost_group = arg_parser.add_argument_group('boost', 'boosting setting')
boost_group.add_argument('--lr_f', default=0.1, type=float, help='lr for weak learner')
boost_group.add_argument('--lr_milestones', default='100,200', type=str, help='lr decay milestones')
boost_group.add_argument('--ensemble_reweight', default="1.0", type=str, help='ensemble weight of early classifiers')
boost_group.add_argument('--loss_equal', action='store_true', help='loss equalization')

# distributed training
boost_group.add_argument('--distributed', action='store_true', help='enables distributed processes')
boost_group.add_argument('--dist_backend', default='nccl', help="dist backend")
boost_group.add_argument('--local_rank', default=0, type=int, help="idx of node")

args = arg_parser.parse_args()

if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
args.scale_list = list(map(int, args.scale_list.split('-')))
args.nScales = len(args.grFactor)

if args.use_valid:
    args.splits = ['train', 'val', 'test']
else:
    args.splits = ['train', 'val']

if args.dataset == 'cifar10':
    args.num_classes = 10
elif args.dataset == 'cifar100':
    args.num_classes = 100
else:
    args.num_classes = 1000

args.lr_milestones = list(map(int, args.lr_milestones.split(',')))

args.ensemble_reweight = list(map(float, args.ensemble_reweight.split(',')))
n_blocks = args.nBlocks * len(args.scale_list) if args.arch == 'ranet' else args.nBlocks
assert len(args.ensemble_reweight) in [1, 2, n_blocks]
if len(args.ensemble_reweight) == 1:
    args.ensemble_reweight = args.ensemble_reweight * n_blocks
elif len(args.ensemble_reweight) == 2:
    args.ensemble_reweight = list(np.linspace(args.ensemble_reweight[0], args.ensemble_reweight[1], n_blocks))