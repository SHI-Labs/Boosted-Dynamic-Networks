import os
import logging
import shutil
import numpy as np

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
from torch.utils.tensorboard import SummaryWriter

from models import msdnet, msdnet_ge, ranet, dynamic_net, dynamic_net_ranet
from op_counter import measure_model
from dataloader import get_dataloaders
from utils.utils import setup_logging
from args import args


def test(model, test_loader):
    model.eval_all()

    n_blocks = args.nBlocks * len(args.scale_list) if args.arch == 'ranet' else args.nBlocks
    corrects = [0] * n_blocks
    totals = [0] * n_blocks
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            outs = model.forward(x)
        for i, out in enumerate(outs):
            corrects[i] += (torch.argmax(out, 1) == y).sum().item()
            totals[i] += y.shape[0]
    return [c / t * 100 for c, t in zip(corrects, totals)]


def log_step(step, name, value, sum_writer, silent=False):
    if not silent:
        logging.info(f'step {step}, {name} {value:.4f}')
    sum_writer.add_scalar(f'{name}', value, step)


def train(model, train_loader, optimizer, epoch, sum_writer):
    model.train_all()
    criterion = torch.nn.CrossEntropyLoss().cuda()

    n_blocks = args.nBlocks * len(args.scale_list) if args.arch == 'ranet' else args.nBlocks
    for it, (x, y) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()
        preds, pred_ensembles = model.forward_all(x, n_blocks - 1)
        loss_all = 0
        for stage in range(n_blocks):
            # train weak learner
            # fix F
            with torch.no_grad():
                if not isinstance(pred_ensembles[stage], torch.Tensor):
                    out = torch.unsqueeze(torch.Tensor([pred_ensembles[stage]]), 0)  # 1x1
                    out = out.expand(x.shape[0], args.num_classes).cuda()
                else:
                    out = pred_ensembles[stage]
                out = out.detach()

            loss = criterion(preds[stage] + out, y)
            if it % 50 == 0:
                log_step(epoch * len(train_loader) + it, f'stage_{stage}_loss', loss, sum_writer)
            loss_all = loss_all + loss

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()


def main():
    torch.backends.cudnn.benchmark = True

    setup_logging(os.path.join(args.result_dir, 'log.txt'))
    logging.info("running arguments: %s", args)
    sum_writer = SummaryWriter(os.path.join(args.result_dir, 'summary'))

    if args.arch == 'msdnet':
        model_func = msdnet
    elif args.arch == 'msdnet_ge':
        model_func = msdnet_ge
    elif args.arch == 'ranet':
        model_func = ranet
    else:
        raise Exception('unknown model name')

    backbone = model_func(args)
    n_flops, n_params = measure_model(backbone, 32, 32)
    torch.save(n_flops, os.path.join(args.result_dir, 'flops.pth'))
    n_blocks = args.nBlocks * len(args.scale_list) if args.arch == 'ranet' else args.nBlocks
    for i in range(n_blocks):
        log_step(i, 'model_size', n_params[i], sum_writer)
        log_step(i, 'model_macs', n_flops[i], sum_writer)
    del(backbone)

    backbone = model_func(args)
    if args.arch == 'ranet':
        model = dynamic_net_ranet(backbone, args).cuda_all()
    else:
        model = dynamic_net(backbone, args).cuda_all()
    train_loader, val_loader, _ = get_dataloaders(args)

    if args.arch != 'ranet':
        args.weight_decay = list(map(float, args.weight_decay.split(',')))
        args.weight_decay = list(np.linspace(args.weight_decay[0], args.weight_decay[-1], n_blocks))
        params_group = []
        for i in range(n_blocks):
            param_i = model.parameters_m(i, separate=False)
            params_group.append({'params': param_i, 'weight_decay': args.weight_decay[i]})
    else:
        args.weight_decay = list(map(float, args.weight_decay.split(',')))
        assert len(args.weight_decay) == 1
        params_group = [{'params': model.parameters_all(n_blocks-1, all_classifiers=True),
                         'weight_decay': args.weight_decay[0]}]  
    optimizer = torch.optim.SGD(params_group, args.lr_f, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_milestones, gamma=0.1)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.result_dir + '/model_latest.pth')
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

    best_accu = -1
    for epoch in range(start_epoch, args.epochs):
        logging.info(f'epoch {epoch}')

        train(model, train_loader, optimizer, epoch, sum_writer)
        scheduler.step()

        accus_test = test(model, val_loader)
        for i, accu in enumerate(accus_test):
            log_step((epoch + 1) * len(train_loader), f'stage_{i}_accu', accu, sum_writer)

        accus_train = test(model, train_loader)
        for i, accu in enumerate(accus_train):
            log_step((epoch + 1) * len(train_loader), f'stage_{i}_accu_train', accu, sum_writer)

        log_step((epoch + 1) * len(train_loader), f'stage_lr', optimizer.param_groups[0]['lr'], sum_writer)

        torch.save(
            {'epoch': epoch,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(),
            },
            args.result_dir + '/model_latest.pth')

        if accus_test[-1] >= best_accu:
            best_accu = accus_test[-1]
            shutil.copyfile(args.result_dir + '/model_latest.pth', args.result_dir + '/model_best.pth')


if __name__ == "__main__":
    main()
