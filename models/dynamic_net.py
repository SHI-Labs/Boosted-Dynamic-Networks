import torch
from utils.distributed import CustomDistributedDataParallel


class DynamicNet(object):
    def __init__(self, model, args):
        self.model = model
        self.nBlocks = args.nBlocks
        self.reweight = args.ensemble_reweight

    def eval(self, stage):
        if isinstance(self.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel, CustomDistributedDataParallel)):
            self.model.module.blocks[stage].eval()
            self.model.module.classifier[stage].eval()
        else:            
            self.model.blocks[stage].eval()
            self.model.classifier[stage].eval()

    def train(self, stage):
        if isinstance(self.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel, CustomDistributedDataParallel)):
            self.model.module.blocks[stage].train()
            self.model.module.classifier[stage].train()
        else:
            self.model.blocks[stage].train()
            self.model.classifier[stage].train()

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

    def parameters_m(self, stage, separate=False):
        if isinstance(self.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel, CustomDistributedDataParallel)):
            model = self.model.module
        else:
            model = self.model

        if separate:
            return list(model.blocks[stage].parameters()), list(model.classifier[stage].parameters())
        else:
            return list(model.blocks[stage].parameters()) + list(model.classifier[stage].parameters())

    def parameters_all(self, stage, all_classifiers=False):
        if isinstance(self.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel, CustomDistributedDataParallel)):
            model = self.model.module
        else:
            model = self.model

        if all_classifiers:
            params = list(model.blocks[:stage + 1].parameters()) + list(model.classifier[:stage + 1].parameters())
        else:
            params = list(model.blocks[:stage + 1].parameters()) + list(model.classifier[stage].parameters())
        return params

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
