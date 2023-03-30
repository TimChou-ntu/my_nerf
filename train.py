import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from collections import defaultdict
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

from model import Positional_Embedding, NeRF
from dataset import dataset_dict
from render import render_rays
from losses import loss_dict
from utils import get_optimizer, get_scheduler, get_learning_rate
from opt import get_opts
from metrics import psnr, ssim
from utils import load_ckpt

class NeRFSystem(LightningModule):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.loss = loss_dict[hparams.loss_type]()

        self.embedding_xyz = Positional_Embedding(10)
        self.embedding_dir = Positional_Embedding(4)
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]

    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        return rays, rgbs
    
    def prepare_data(self) -> None:
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {
            'root_dir': self.hparams.root_dir,
            'img_wh': tuple(self.hparams.img_wh)
        }
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)


    def forward(self, rays):
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            render_ray_chunk = render_rays(
                self.models,
                self.embeddings,
                rays[i:i+self.hparams.chunk],
                self.hparams.N_samples,
                self.hparams.use_disp,
                self.hparams.perturb,
                self.hparams.noise_std,
                self.hparams.N_importance,
                self.hparams.chunk,
                self.train_dataset.white_back
            )
            for k, v in render_ray_chunk.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        return results

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        self.scheduler = get_scheduler(self.hparams, self.optimizer)

        return [self.optimizer], [self.scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, num_workers=4, batch_size=self.hparams.batch_size, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, num_workers=4, batch_size=1, pin_memory=True) # validate one image at a time

    def test_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, num_workers=4, batch_size=1, pin_memory=True) # validate one image at a time
   
    def training_step(self, batch, batch_nb):
        self.log("lr", get_learning_rate(self.optimizer))
        rays, rgbs = self.decode_batch(batch)
        results = self(rays)
        loss = self.loss(results, rgbs)
        self.log("loss", loss)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            self.log('train/psnr', psnr_)

        return {
            'loss': loss,
            'progress_bar': {'train_psnr': psnr_},
            'log': self.log
        }
    
    def validation_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        # Not sure why squeeze
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze()
        results = self(rays)
        loss = self.loss(results, rgbs)
        self.log('val/loss', loss)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        self.log('val/psnr', psnr_)
        
        return {'val/loss': loss}
    
    def on_test_start(self):
        for i, model in enumerate(self.models):
            if i == 0:
                load_ckpt(model, self.hparams.nerf_ckpt, model_name='nerf_coarse')
                print(f"load nerf_coarse from {self.hparams.nerf_ckpt}")
            elif i == 1:
                load_ckpt(model, self.hparams.nerf_ckpt, model_name='nerf_fine')
                print(f"load nerf_fine from {self.hparams.nerf_ckpt}")
        return
    
    def visualize_depth(self, depth, cmap=cv2.COLORMAP_JET):
        """
        depth: (H, W)
        """
        x = depth.cpu().numpy()
        x = np.nan_to_num(x) # change nan to 0
        mi = np.min(x) # get minimum depth
        ma = np.max(x)
        x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
        x = (255*x).astype(np.uint8)
        x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
        x_ = T.ToTensor()(x_) # (3, H, W)
        return x_

    def test_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        # Not sure why squeeze
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze()
        results = self(rays)
        loss = self.loss(results, rgbs)
        print('test/loss', loss)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        print('test/psnr', psnr_)

        img_gt = batch['rgbs'].view(self.hparams.img_wh[1], self.hparams.img_wh[0], 3).cpu().numpy()
        img_pred = results['rgb_fine'].view(self.hparams.img_wh[1], self.hparams.img_wh[0], 3).cpu().numpy()
        alpha_pred = results['opacity_fine'].view(self.hparams.img_wh[1], self.hparams.img_wh[0]).cpu().numpy()
        depth_pred = results['depth_fine'].view(self.hparams.img_wh[1], self.hparams.img_wh[0])

        plt.subplots(figsize=(15, 8))
        plt.tight_layout()
        plt.subplot(221)
        plt.title('GT')
        plt.imshow(img_gt)
        plt.subplot(222)
        plt.title('pred')
        plt.imshow(img_pred)
        plt.subplot(223)
        plt.title('depth')
        plt.imshow(self.visualize_depth(depth_pred).permute(1,2,0))
        plt.subplot(224)
        plt.title('alpha')
        plt.imshow(alpha_pred, cmap='gray')
        path = f"./results/{self.hparams.exp_name}/"
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f'{path}{batch_nb}.png')

        return {'test/loss': loss}
    
if __name__ == '__main__':
    seed_everything(5)
    hparams = get_opts()
    system = NeRFSystem(hparams)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(f'ckpts/{hparams.exp_name}'),
        monitor='val/loss',
        save_top_k=5
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="./",version=hparams.exp_name)

    trainer = Trainer(
        max_epochs=hparams.num_epochs,
        callbacks=checkpoint_callback,
        logger=tb_logger,
        accelerator="gpu", 
        devices=1,
        precision=16
    )

    if hparams.mode == 'train':
        trainer.fit(system)
    elif hparams.mode == 'test':
        trainer.test(system)
    else:
        raise ValueError(f"Unknown mode {hparams.mode}")