import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.backends.cudnn as cudnn

from models import msdnet, msdnet_ge, ranet, dynamic_net, dynamic_net_ranet
from args import args
from dataloader import get_dataloaders
from adaptive_inference import dynamic_evaluate

from op_counter import measure_model
from utils.utils import AverageMeter, accuracy
from utils.utils import CustomizedOpen


def validate(val_loader, model):
    top1, top5 = [], []
    n_blocks = args.nBlocks * len(args.scale_list) if args.arch == 'ranet' else args.nBlocks
    for i in range(n_blocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    model.eval_all()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model.forward(input_var)
            if not isinstance(output, list):
                output = [output]

            for j in range(len(output)):
                prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                top1[j].update(prec1.item(), input.size(0))
                top5[j].update(prec5.item(), input.size(0))

            if i % args.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Acc@1 {top1.val:.4f}\t'
                      'Acc@5 {top5.val:.4f}'.format(
                        i + 1, len(val_loader), top1=top1[-1], top5=top5[-1]))
    n_blocks = args.nBlocks * len(args.scale_list) if args.arch == 'ranet' else args.nBlocks
    for j in range(n_blocks):
        print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[j], top5=top5[j]))
    return [_.avg for _ in top1], [_.avg for _ in top5]


def main():
    if args.arch == 'msdnet':
        model_func = msdnet
    elif args.arch == 'msdnet_ge':
        model_func = msdnet_ge
    elif args.arch == 'ranet':
        model_func = ranet
    else:
        raise Exception('unknown model name')

    backbone = model_func(args)
    n_flops, n_params = measure_model(backbone, 224, 224)
    if args.arch == 'ranet':
        model = dynamic_net_ranet(backbone, args).cuda_all()
    else:
        model = dynamic_net(backbone, args).cuda_all()

    cudnn.benchmark = True
    _, val_loader, test_loader = get_dataloaders(args)
    data_loader = test_loader

    state_dict = torch.load(args.evaluate_from)['state_dict']
    model.load_state_dict(state_dict)

    if args.evalmode == 'anytime':
        save_path = args.evaluate_from.split('model_best')[0] + 'any{}.txt'.format(args.save_suffix)
        top1, top5 = validate(data_loader, model)
        with CustomizedOpen(save_path, 'w') as f:
            for a, b, c, d in zip(n_flops, n_params, top1, top5):
                f.write('{} {} {} {}\n'.format(a, b, c, d))
    else:
        dynamic_evaluate(model, data_loader, val_loader, args)


if __name__ == '__main__':
    main()
