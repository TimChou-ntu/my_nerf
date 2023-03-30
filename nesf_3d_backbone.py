import torch
import os
import numpy as np
from model import *

from utils import load_ckpt
import metrics

from dataset import dataset_dict
from depth_utils import *

import mcubes
import trimesh

# lego/fern/nesf0/nesf1
scene_name = 'nesf1'

torch.backends.cudnn.benchmark = True

def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    # if ndc:
    #     rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs

def _compute_bbox_by_cam_frustrm_bounded(cfg, HW, Ks, poses, i_train, near, far):
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    return xyz_min, xyz_max

def compute_bbox_by_cam_frustrm_bounded(dataset, near, far):
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for i, data in enumerate(dataset):
        rays_o, rays_d = data['rays'][:3], data['rays'][3:6]
        pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0)))

    return xyz_min, xyz_max


# Change here #
if scene_name == 'lego':
    img_wh = (800, 800) # full resolution of the input images
    dataset_name = 'blender' # blender or llff (own data)
    root_dir = '/home/timothy/Desktop/2022Fall/nerf_pl/nerf_synthetic/lego' # the folder containing data
    ckpt_path = '/home/timothy/Desktop/2023Spring/my_nerf/ckpts/exp_white_back_false/epoch=15-step=250000.ckpt' # the model path

elif scene_name == 'fern':
    img_wh = (504, 378) # full resolution of the input images
    dataset_name = 'llff' # blender or llff (own data)
    root_dir = '/home/timothy/Desktop/2023Spring/my_nerf/nerf_llff_data/fern' # the folder containing data
    ckpt_path = '/home/timothy/Desktop/2023Spring/my_nerf/ckpts/exp/epoch=29-step=106050.ckpt' # the model path

elif scene_name == 'nesf0':
    img_wh = (256, 256) # full resolution of the input images
    dataset_name = 'klevr' # blender or llff (own data)
    root_dir = '/home/timothy/Desktop/2023Spring/my_nerf/nesf_dataset/klevr/0' # the folder containing data
    ckpt_path = '/home/timothy/Desktop/2023Spring/my_nerf/ckpts/exp_nesf/epoch=15-step=295936.ckpt' # the model path

elif scene_name == 'nesf1':
    img_wh = (256, 256) # full resolution of the input images
    dataset_name = 'klevr' # blender or llff (own data)
    root_dir = '/home/timothy/Desktop/2023Spring/GeoNeRF/data/data/nesf_data/klevr/1' # the folder containing data
    # ckpt_path = "/home/timothy/Desktop/2023Spring/my_nerf/ckpts/nesf1/epoch=15-step=215040.ckpt" # the model path
    # ckpt_path = "/home/timothy/Desktop/2023Spring/my_nerf/ckpts/nesf1_calculate_near_far_revise/epoch=15-step=215040.ckpt"
    ckpt_path = "/home/timothy/Desktop/2023Spring/my_nerf/ckpts/nesf1_white_back_false/epoch=14-step=201600.ckpt" # the model path

else:
    raise NotImplementedError("Currently only lego/fern/nesf0")
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

# xyz_min, xyz_max = compute_bbox_by_cam_frustrm_bounded(dataset, dataset.near, dataset.far)
# print(xyz_min, xyz_max)
# '''

embedding_xyz = Positional_Embedding(10)
embedding_dir = Positional_Embedding(4)

nerf_fine = NeRF()
load_ckpt(nerf_fine, ckpt_path, model_name='nerf_fine')

# nerf_system = NeRFSystem.load_from_checkpoint(ckpt_path,hparams_file=hparam_path)
# nerf_fine = nerf_system.nerf_fine

nerf_fine.cuda().eval()
# "scene_boundaries": {"min": [-3.1, -3.1, -0.1], "max": [3.1, 3.1, 3.1]}
### Tune these parameters until the whole object lies tightly in range with little noise ###
N = 128 # controls the resolution, set this number small here because we're only finding
        # good ranges here, not yet for mesh reconstruction; we can set this number high
        # when it comes to final reconstruction.
xmin, xmax = -1, 1 # left/right range
ymin, ymax = -1, 1 # forward/backward range
# zmin, zmax = -1.2, 1.2 # up/down range
zmin, zmax = -1, 1 # up/down range
## Attention! the ranges MUST have the same length!
sigma_threshold = 30. # controls the noise (lower=maybe more noise; higher=some mesh might be missing)
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
sigma = sigma.reshape(N, N, N).transpose(1, 0, 2)

# The below lines are for visualization, COMMENT OUT once you find the best range and increase N!
vertices, triangles = mcubes.marching_cubes(sigma, sigma_threshold)
mesh = trimesh.Trimesh(vertices/N, triangles)
mesh.show()
# '''