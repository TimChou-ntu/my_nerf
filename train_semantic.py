import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything

from model import Positional_Embedding, NeRF, Semantic_MLP
from dataset import dataset_dict
from render import render_rays, render_semantic_rays
from losses import loss_dict
from utils import get_optimizer, get_scheduler, get_learning_rate, load_ckpt
from opt import get_opts
from metrics import psnr, ssim
from UNet import UNet


class SemanticSystem(LightningModule):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.loss = loss_dict[hparams.loss_type]()

        self.embedding_xyz = Positional_Embedding(10)
        self.embedding_dir = Positional_Embedding(4)
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        self.nerf = NeRF()
        # only these two models should update weights
        self.semantic_mlp = Semantic_MLP(32, self.hparams.semantic_class)
        self.UNet = UNet(out_dim=32)

        # resolution should be single integer
        self.resolution = hparams.resolution
        self.nerf_ckpts = hparams.nerf_ckpt
        self.chunk = hparams.chunk

        # maybe load ckpt when doing inference instead of load it first
        # load_ckpt(self.nerf, hparams.Nerf_ckpt, model_name='nerf_fine')

    def prepare_data(self) -> None:
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {
            'root_dir': self.hparams.root_dir,
            'img_wh': tuple(self.hparams.img_wh),
        }
        if self.hparams.dataset_name == 'klevr':
            kwargs['get_rgb'] = False
            kwargs['get_semantic'] = True
        else:
            raise NotImplementedError("Semantic should only train on Klevr dataset")
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def sigma_grid(self):
        load_ckpt(self.nerf, self.nerf_ckpts, model_name='nerf_fine')
        self.nerf.cuda().eval()
        indice = np.linspace(-1, 1, self.resolution)
        xyz_ = torch.FloatTensor(np.stack(np.meshgrid(indice, indice, indice), -1).reshape(-1, 3)).cuda()
        dir_ = torch.zeros_like(xyz_).cuda()
        with torch.no_grad():
            B = xyz_.shape[0]
            out_chunks = []
            for i in range(0, B, self.chunk):
                xyz_embedded = self.embedding_xyz(xyz_[i:i+self.chunk])
                dir_embedded = self.embedding_dir(dir_[i:i+self.chunk])
                xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded], 1)
                out_chunks += [self.nerf(xyzdir_embedded)]
            rgbsigma = torch.cat(out_chunks, 0)
        # non-negative sigma grid
        sigma = torch.maximum(rgbsigma[:, -1], torch.zeros_like(rgbsigma[:, -1]))
        sigma = sigma.reshape((self.resolution,self.resolution,self.resolution))
        return sigma

    def decode_batch(self, batch):
        rays = batch['rays']
        # rgbs = batch['rgbs']
        semantics = batch['semantics']
        return rays, semantics
    
    def forward(self, rays):
        sigma = self.sigma_grid().unsqueeze(0).unsqueeze(0) # (1, 1, resolution, resolution, resolution)
        feature_grid = self.UNet(sigma) # (1, 11, resolution, resolution, resolution)
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0,B, self.hparams.chunk):
            render_rays_chunk = render_semantic_rays(
                self.nerf,
                self.semantic_mlp,
                feature_grid,
                self.embeddings,
                rays[i:i+self.hparams.chunk],
                self.hparams.N_samples,
                self.hparams.use_disp,
                self.hparams.perturb,
                self.hparams.noise_std,
                self.hparams.N_importance,
                self.hparams.chunk,
                self.train_dataset.white_back,
            )
            for k, v in render_rays_chunk.items():
                results[k] += [v]
        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        
        return results

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, [self.UNet, self.semantic_mlp])
        self.scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [self.scheduler]
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, num_workers=4, batch_size=self.hparams.batch_size, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, num_workers=4, batch_size=1, pin_memory=True) # validate one image at a time
    
    def training_step(self, batch, batch_nb):
        # TODO do forward and loss
        self.log("lr", get_learning_rate(self.optimizer))
        rays, semantics = self.decode_batch(batch)
        results = self(rays)
        loss = self.loss(results, semantics)
        self.log("loss", loss)


        return {
            'loss': loss,
            'progress_bar': {"train_loss": loss},
            'log': self.log
        }
    def validation_step(self, batch, batch_nb):
        rays, semantics = self.decode_batch(batch)
        rays = rays.squeeze()
        semantics = semantics.squeeze()
        results = self(rays)
        loss = self.loss(results, semantics)
        self.log("val/loss", loss)

        return {'val/loss': loss}
    
if __name__ == '__main__':
    seed_everything(5)
    hparams = get_opts()
    system = SemanticSystem(hparams)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(f'semantic_ckpts/{hparams.exp_name}'),
        monitor='val/loss',
        save_top_k=5,
    )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir='./semantic_logs', name=hparams.exp_name)
    trainer = Trainer(
        max_epochs=hparams.num_epochs,
        callbacks=checkpoint_callback,
        logger=tb_logger,
        accelerator='gpu',
        devices=1,
        precision=16,
    )
    trainer.fit(system)