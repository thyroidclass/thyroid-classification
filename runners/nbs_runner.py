import torch
from torch.distributions.exponential import Exponential

import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path

from utils.metrics import Accuracy, ConfusionMatrix
from runners.base_runner import gather_tensor
from runners.cnn_runner import CnnRunner

class NbsRunner(CnnRunner):
    def __init__(self, loader, model, optim, lr_scheduler, num_epoch,
                 loss_with_weight, val_metric, test_metric, logger,
                 model_path, infer_path, rank, epoch_th, num_mc):

        self.num_mc = num_mc
        self.n_a = loader.n_a
        self.epoch_th = epoch_th
        self.alpha = torch.ones([1, self.n_a])

        super().__init__(loader, model, optim, lr_scheduler, num_epoch, loss_with_weight,
                         val_metric, test_metric, logger, model_path, infer_path, rank)

        self.save_kwargs['alpha'] = self.alpha
        self._update_weight()

    def _update_weight(self):
        if self.epoch > self.epoch_th:
            self.alpha = Exponential(torch.ones([1, self.n_a])).sample()

    def _calc_loss(self, img, label, idx):
        n0 = img.size(0)
        w = self.alpha[0, idx].cuda()
        
        if len(img.size()) == 5:
            img = img.flatten(start_dim=0, end_dim=1).cuda(non_blocking=True)
        else:
            img = img.cuda(non_blocking=True)

        output = self.model(img, self.alpha.repeat_interleave(n0, 0))
        for _ in range(output.dim() - w.dim()):
            w.unsqueeze_(-1)
        label = label.cuda(non_blocking=True)
        
        if (label.size() != output.size()):
            label = label.float().unsqueeze(1)
            output = output.float()

        loss_ = 0
        for loss, _w in self.loss_with_weight:
            _loss = _w * loss(output, label, w)
            loss_ += _loss

        return loss_

    @torch.no_grad()
    def _valid_a_batch(self, img, label, with_output=False):
        self._update_weight()
        self.model.eval()

        if len(img.size()) == 5:
            img = img.flatten(start_dim=0, end_dim=1).cuda(non_blocking=True)
        else:
            img = img.cuda(non_blocking=True)

        print(img.shape)
        output = self.model(img, self.num_mc)
        print(output.shape)
        label = label.cuda(non_blocking=True)
        output = output.sigmoid()
        result = self.val_metric(output, label)
        if with_output:
            result = [result, output]
        return result

    def test(self):
        self.load('model.pth')
        loader = self.loader.load('test')

        if self.rank == 0:
            t_iter = tqdm(loader, total=self.loader.len)
        else:
            t_iter = loader

        paths = []
        coords_i = []
        coords_j = []
        outputs = []
        labels = []
        metrics = []
        self.model.eval()

        for path, i, j, img, label in t_iter:
            _metric, output = self._valid_a_batch(img, label, with_output=True)
            paths += path
            coords_i += i
            coords_j += j 
            labels += [gather_tensor(label).cpu().numpy()]
            outputs += [gather_tensor(output).cpu().numpy()]
            metrics += [gather_tensor(_metric).cpu().numpy()]

        coords = np.concatenate(([coords_i], [coords_j]), 0).transpose(1,0)
        labels = np.concatenate(labels)
        outputs = np.concatenate(outputs, axis=1)

        acc = Accuracy(nlabels=1)(outputs, labels) * 100
        cm = ConfusionMatrix(nlabels=1)(outputs, labels)[0]

        log = f'[Test] Acc: {acc:.2f}, TN: {cm[1].item()}, FP:{cm[2].item()}, FN:{cm[3].item()}, TP: {cm[0].item()}'
        self.log(log, 'info')

        encoded_paths = []
        for p in paths:
            encoded_paths.append(p.encode())
        with h5py.File(f"{self.model_path}/output.h5", 'w') as h:
            h.create_dataset('paths', data=encoded_paths)
            h.create_dataset('coords', data=coords)
            h.create_dataset('outputs', data=outputs.transpose(1,0,2).squeeze())
            h.create_dataset('labels', data=labels)
