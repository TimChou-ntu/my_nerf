import os
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything

from model import Positional_Embedding, NeRF
from dataset import dataset_dict
from render import render_rays
from losses import loss_dict
from utils import get_optimizer, get_scheduler, get_learning_rate
from opt import get_opts
from metrics import psnr, ssim


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


    
if __name__ == '__main__':
    seed_everything(5)
    hparams = get_opts()
    system = NeRFSystem(hparams)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(f'ckpts/{hparams.exp_name}'),
        monitor='val/loss',
        save_top_k=5
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="./")

    trainer = Trainer(
        max_epochs=hparams.num_epochs,
        callbacks=checkpoint_callback,
        logger=tb_logger,
        accelerator="gpu", 
        devices=1,
        precision=16
    )

    trainer.fit(system)