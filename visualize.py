import torch
from collections import defaultdict
import numpy as np
import mcubes
import trimesh

from render import *
from model import *
from utils import *
from dataset import dataset_dict
from train import NeRFSystem
from opt import get_opts

# Change here #
img_wh = (800, 800) # full resolution of the input images
dataset_name = 'blender' # blender or llff (own data)
scene_name = 'lego' # whatever you want
root_dir = '/home/timothy/Desktop/2022Fall/nerf_pl/nerf_synthetic/lego' # the folder containing data
ckpt_path = '/home/timothy/Desktop/2023Spring/my_nerf/ckpts/exp_white_back_false/epoch=15-step=250000.ckpt' # the model path
hparam_path = '/home/timothy/Desktop/2023Spring/my_nerf/lightning_logs/version_11/hparams.yaml'
###############

kwargs = {'root_dir': root_dir,
          'img_wh': img_wh}
if dataset_name == 'llff':
    kwargs['spheric_poses'] = True
    kwargs['split'] = 'test'
else:
    kwargs['split'] = 'train'
    
chunk = 1024*32
dataset = dataset_dict[dataset_name](**kwargs)

embedding_xyz = Positional_Embedding(10)
embedding_dir = Positional_Embedding(4)

nerf_fine = NeRF()
load_ckpt(nerf_fine, ckpt_path, model_name='nerf_fine')

# nerf_system = NeRFSystem.load_from_checkpoint(ckpt_path,hparams_file=hparam_path)
# nerf_fine = nerf_system.nerf_fine

nerf_fine.cuda().eval()

### Tune these parameters until the whole object lies tightly in range with little noise ###
N = 128 # controls the resolution, set this number small here because we're only finding
        # good ranges here, not yet for mesh reconstruction; we can set this number high
        # when it comes to final reconstruction.
xmin, xmax = -1.2, 1.2 # left/right range
ymin, ymax = -1.2, 1.2 # forward/backward range
# zmin, zmax = -1.2, 1.2 # up/down range
zmin, zmax = -0.6, 1.8 # up/down range
## Attention! the ranges MUST have the same length!
sigma_threshold = 70. # controls the noise (lower=maybe more noise; higher=some mesh might be missing)
############################################################################################

x = np.linspace(xmin, xmax, N)
y = np.linspace(ymin, ymax, N)
z = np.linspace(zmin, zmax, N)

xyz_ = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3)).cuda()
dir_ = torch.zeros_like(xyz_).cuda()

with torch.no_grad():
    B = xyz_.shape[0]
    out_chunks = []
    for i in range(0, B, chunk):
        xyz_embedded = embedding_xyz(xyz_[i:i+chunk]) # (N, embed_xyz_channels)
        dir_embedded = embedding_dir(dir_[i:i+chunk]) # (N, embed_dir_channels)
        xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded], 1)
        out_chunks += [nerf_fine(xyzdir_embedded)]
    rgbsigma = torch.cat(out_chunks, 0)
    
sigma = rgbsigma[:, -1].cpu().numpy()
sigma = np.maximum(sigma, 0)
sigma = sigma.reshape(N, N, N)

# The below lines are for visualization, COMMENT OUT once you find the best range and increase N!
vertices, triangles = mcubes.marching_cubes(sigma, sigma_threshold)
mesh = trimesh.Trimesh(vertices/N, triangles)
mesh.show()