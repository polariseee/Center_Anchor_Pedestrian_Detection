import os
import torch
import torch.nn as nn

from mmcv.utils import get_logger
from models.detectors import SingleStageDetector
from models.py_utils.data_parallel import DataParallel
import pdb

torch.manual_seed(317)  # 设置随机数种子

class Network(nn.Module):
    def __init__(self, model, loss):
        super(Network, self).__init__()

        self.model = model
        self.loss  = loss

    def forward(self, xs, ys, img_meta, **kwargs):
        preds = self.model(*xs, **kwargs)
        loss  = self.loss(preds, ys, img_meta, **kwargs)
        return loss

# for model backward compatibility
# previously model was wrapped by DataParallel module
class DummyModule(nn.Module):
    def __init__(self, model):
        super(DummyModule, self).__init__()
        self.module = model

    def forward(self, *xs, **kwargs):
        return self.module(*xs, **kwargs)

class NetworkFactory(object):
    def __init__(self, cfg):
        # super(NetworkFactory, self).__init__()
        self.train_cfg = cfg.train_cfg
        self.test_cfg = cfg.test_cfg
        nnet_module = SingleStageDetector(cfg)

        self.model   = DummyModule(nnet_module)
        self.loss    = nnet_module.loss
        self.network = Network(self.model, self.loss)
        # self.network = DataParallel(self.network, chunk_sizes=cfg.train_cfg.chunk_sizes)

        total_params = 0
        for params in self.model.parameters():
            num_params = 1
            for x in params.size():
                num_params *= x
            total_params += num_params
        print("total parameters: {}".format(total_params))

        if cfg.train_cfg.opt_algo == "adam":
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters())
            )
        elif cfg.train_cfg.opt_algo == "sgd":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=cfg.train_cfg.learning_rate,
                momentum=0.9, weight_decay=0.0001
            )
        else:
            raise ValueError("unknown optimizer")

    def cuda(self):
        self.model.cuda()
        self.teacher_dict = self.model.state_dict()

    def train_mode(self):
        self.network.train()

    def eval_mode(self):
        self.network.eval()

    def train(self, xs, ys, img_meta, **kwargs):
        xs = [x.cuda(non_blocking=True) for x in xs]
        ys = [y.cuda(non_blocking=True) for y in ys]

        self.optimizer.zero_grad()
        losses = self.network(xs, ys, img_meta)

        if isinstance(losses, tuple):
            loss = losses[0] + losses[1]
        else:
            loss = losses

        loss.backward()
        self.optimizer.step()

        for k, v in self.model.state_dict().items():
            if k.find('num_batches_tracked') == -1:
                self.teacher_dict[k] = self.train_cfg.alpha * self.teacher_dict[k]\
                                       + (1 - self.train_cfg.alpha) * v
            else:
                self.teacher_dict[k] = 1 * v

        return losses

    def validate(self, xs, ys, **kwargs):
        with torch.no_grad():
            xs = [x.cuda(non_blocking=True) for x in xs]
            ys = [y.cuda(non_blocking=True) for y in ys]

            loss = self.network(xs, ys)
            loss = loss.mean()
            return loss

    def test(self, xs, **kwargs):
        with torch.no_grad():
            xs = [x.cuda(non_blocking=True) for x in xs]
            return self.model(*xs, **kwargs)

    def set_lr(self, lr):
        print("setting learning rate to: {}".format(lr))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def load_pretrained_params(self, pretrained_model):
        print("loading from {}".format(pretrained_model))
        with open(pretrained_model, "rb") as f:
            params = torch.load(f)
            self.model.load_state_dict(params)

    def load_params(self, epoch):
        cache_file = os.path.join(self.train_cfg.work_dir, 'ckpt', 'epoch_{}.pth'.format(epoch))
        print("loading model from {}".format(cache_file))
        # with open(cache_file, "rb") as f:
        #     params = torch.load(f)
        #     self.model.load_state_dict(params)
        params = torch.load(cache_file)
        self.model.load_state_dict(params)
        if self.test.cfg.test:
            teacher_dict = torch.load(cache_file + '.tea')
            self.model.load_state_dict(teacher_dict)

    def save_params(self, epoch):
        cache_file = os.path.join(self.train_cfg.work_dir, 'ckpt', 'epoch_{}.pth'.format(epoch))
        print("saving model to {}".format(cache_file))
        # with open(cache_file, "wb") as f:
        #     params = self.model.state_dict()
        #     torch.save(params, f)
        torch.save(self.model.state_dict(), cache_file)
        torch.save(self.teacher_dict, cache_file + '.tea')
