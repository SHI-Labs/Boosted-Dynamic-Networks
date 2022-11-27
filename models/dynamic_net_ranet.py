import torch
from utils.distributed import CustomDistributedDataParallel


class DynamicNet(object):
    def __init__(self, model, args):
        self.model = model
        self.nBlocks = args.nBlocks * len(args.scale_list) if args.arch == 'ranet' else args.nBlocks
        self.reweight = args.ensemble_reweight

    def cuda_all(self):
        self.model.cuda()
        return self

    def cpu_all(self):
        self.model.cpu()
        return self

    def eval_all(self):
        self.model.eval()
        return self

    def train_all(self):
        self.model.train()
        return self

    def parameters_all(self, stage, all_classifiers=False):
        if isinstance(self.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel, CustomDistributedDataParallel)):
            model = self.model.module
        else:
            model = self.model

        params = []
        params.extend(list(model.FirstLayer.parameters()))
        params_class = []
        cls_idx = 0
        for ii in range(model.nScales):
            n_block_curr = 0
            for i in range(model.nBlocks[ii+1]):
                params.extend(list(model.scale_flows[ii][i].parameters()))
                n_block_curr += 1
                if n_block_curr > model.nBlocks[ii]:
                    params_class.extend(list(model.classifier[cls_idx].parameters()))
                    if cls_idx == stage:
                        if all_classifiers:
                            return params + params_class
                        else:
                            return params + params_class[-1:]
                    cls_idx += 1

        raise Exception('the parameter stage ({}) should be wrong. please double check.'.format(stage))

    def state_dict(self):
        state_dict = {'model': self.model.state_dict()}
        return state_dict

    def load_state_dict(self, ckpt):
        ckpt = ckpt['model']
        if not hasattr(self.model, 'module'):
            ckpt = {k.split('module.')[-1] if k.startswith('module.') else k: v for k, v in ckpt.items()}
        self.model.load_state_dict(ckpt)

    def forward(self, x):
        outs = self.model(x, self.nBlocks)
        preds = [0]
        for i in range(len(outs)):
            pred = outs[i] + preds[-1] * self.reweight[i]
            preds.append(pred)
        preds = preds[1:]
        return preds

    def forward_all(self, x, stage):
        """Forward the model until block `stage` and get a list of ensemble predictions
        """
        assert 0 <= stage < self.nBlocks
        outs = self.model(x, stage)
        preds = [0]
        for i in range(len(outs)):
            pred = (outs[i] + preds[-1]) * self.reweight[i]
            preds.append(pred)
            if i == stage:
                break
        return outs, preds
