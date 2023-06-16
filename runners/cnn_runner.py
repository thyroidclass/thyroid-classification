import torch

import h5py
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from time import time

from utils.metrics import Accuracy, ConfusionMatrix
from runners.base_runner import BaseRunner, reduce_tensor, gather_tensor


class CnnRunner(BaseRunner):
    def __init__(self, loader, model, optim, lr_scheduler, num_epoch, loss_with_weight,
                 val_metric, test_metric, logger, model_path, infer_path, rank):
        self.num_epoch = num_epoch
        self.epoch = 0
        self.loss_with_weight = loss_with_weight
        self.val_metric = val_metric
        self.test_metric = test_metric
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.best_score = 0.
        self.save_kwargs = {}
        self.world_size = torch.distributed.get_world_size()
        self.infer_path = Path(infer_path)
        super().__init__(loader, model, logger, model_path, rank)
        self.load()

    def _calc_loss(self, img, label):
        self.model.train()
        output = self.model(img.cuda(non_blocking=True))
        label = label.cuda(non_blocking=True)

        if (label.size() != output.size()):
            label = label.float().unsqueeze(1)
            output = output.float()

        loss_ = 0
        for loss, w in self.loss_with_weight:
            _loss = w * loss(output, label)
            loss_ += _loss
        return loss_

    def fgsm(self, img, label):
        step_size = 0.01
        loss_fn = self.loss_with_weight[0][0]
        img = img.cuda()
        img.requires_grad = True
        self.model.eval()
        self.model.zero_grad()
        output = self.model(img)
        loss = loss_fn(output, label.cuda())
        loss.backward()
        grad_sign = img.grad.sign()
        img_new = img + step_size * grad_sign
        return img_new.cpu().detach()

    def _train_a_batch(self, *batch):
        loss = self._calc_loss(*batch)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        _loss = reduce_tensor(loss, True).item()
        return _loss

    @torch.no_grad()
    def _valid_a_batch(self, img, label, with_output=False):
        output = self.model(img.cuda(non_blocking=True))
        label = label.cuda(non_blocking=True)
        output = output.sigmoid()
        result = self.val_metric(output, label)
        if with_output:
            result = [result, output]
        return result

    def train(self):
        self.log("Start to train", 'debug')
        for epoch in range(self.epoch, self.num_epoch):
            self.model.train()
            loader = self.loader.load("train")
            if self.rank == 0:
                t_iter = tqdm(loader, total=len(loader),
                              desc=f"[Train {epoch}]")
            else:
                t_iter = loader
            losses = 0
            times = []
            for i, batch in enumerate(t_iter):
                t = time()
                loss = self._train_a_batch(*batch)
                times += [time() - t]
                losses += loss
                if self.rank == 0:
                    t_iter.set_postfix(loss=f"{loss:.4} / {losses/(i+1):.4}")
            self.log(f"[Train] epoch:{epoch} loss:{losses/(i+1)}", 'info')
 
            print("Batch Training Time : ", np.mean(times))
            self.lr_scheduler.step()
            self.val(epoch, (losses / (i + 1)))

    def val(self, epoch, loss):
        loader = self.loader.load('val')
        v_iter = loader

        metrics = []
        self.model.eval()
        for batch in v_iter:
            img, label = batch[:2]
            _metric = self._valid_a_batch(img, label, with_output=False)
            metrics += list(gather_tensor(_metric).cpu().numpy().reshape(-1))
        
        acc = np.mean(metrics) * 100
        self.log(f"[Val] {epoch} Score: {acc}", 'info')

        if self.rank == 0:
            self.save(epoch, acc, **self.save_kwargs)

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
        labels = []
        outputs = []
        metrics = []
        self.model.eval()

        for batch in t_iter:
            path, i, j, img, label = batch[:5]
            _metric, output = self._valid_a_batch(img, label, with_output=True)
            
            paths += path
            coords_i += i
            coords_j += j
            labels += [gather_tensor(label).cpu().numpy()]
            outputs += [gather_tensor(output).cpu().numpy()]
            metrics += [gather_tensor(_metric).cpu().numpy()]

        coords = np.concatenate(([coords_i], [coords_j]), 0).transpose(1, 0)
        labels = np.concatenate(labels)
        outputs = np.concatenate(outputs, axis=0)

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

    def save(self, epoch, metric, file_name="model", **kwargs):
        torch.save({"epoch": epoch,
                    "param": self.model.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "score": metric,
                    "best": self.best_score,
                    "lr_schdlr": self.lr_scheduler.state_dict(),
                    **kwargs}, f"{self.model_path}/{file_name}.pth")

        cond = metric >= self.best_score
        if cond:
            self.log(f"{self.best_score} -------------------> {metric}", 'debug')
            self.best_score = metric
            shutil.copy2(f"{self.model_path}/{file_name}.pth",
                         f"{self.model_path}/best.pth")
            self.log(f"Model has saved at {epoch} epoch.", 'debug')

    def load(self, file_name="model.pth"):
        self.log(self.model_path, 'debug')
        
        load_path = self.model_path

        if (load_path / file_name).exists():
            self.log(f"Loading {load_path} File", 'debug')
            ckpoint = torch.load(f"{load_path}/{file_name}", map_location='cpu')

            for key, value in ckpoint.items():
                if key == 'param': 
                    self.model.load_state_dict(value)
                elif key == 'optimizer':
                    self.optim.load_state_dict(value)
                elif key == 'lr_schdlr':
                    self.lr_scheduler.load_state_dict(value)
                elif key == 'epoch':
                    self.epoch = value + 1
                elif key == 'best':
                    self.best_score = value
                else:
                    self.__dict__[key] = value

            self.log(f"Model Type : {file_name}, epoch : {self.epoch}", 'debug')
        else:
            self.log("Failed to load, not existing file", 'debug')

    def get_lr(self):
        return self.lr_scheduler.optimizer.param_groups[0]['lr']
