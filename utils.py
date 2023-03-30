import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import torch.nn as nn
import torch.nn.functional as F

def get_optimizer(hparams, models):
    eps = 1e-8
    parameters = []
    for model in models:
        parameters += list(model.parameters())
    if hparams.optimizer == 'sgd':
        optimizer = SGD(parameters, lr=hparams.lr, momentum=hparams.momentum, weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'adam':
        optimizer = Adam(parameters, lr=hparams.lr, eps=eps, weight_decay=hparams.weight_decay)
    else:
        raise ValueError('optimizer now only support SGD/Adam')
    
    return optimizer

def get_scheduler(hparams, optimizer):
    eps = 1e-8
    if hparams.lr_scheduler == 'steplr':
        scheduler = MultiStepLR(optimizer, milestones=hparams.decay_step, gamma=hparams.decay_gamma)
    elif hparams.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=hparams.num_epochs, eta_min=eps)
    # elif hparams.warmup_epochs > 0 and hparams.optimizer not in ['radam', 'ranger']: # if setting warmup_epochs then do warmup scheduler
    #     scheduler = GradualWarmupScheduler()
    else:
        raise ValueError('scheduler now only support steplr/cosine')

    return scheduler

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                print('ignore', k)
                break
        else:
            checkpoint_[k] = v
    return checkpoint_

def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)


def trilinear_interpolation(features, points):
    '''
    Args: features: (dim, N, N, N)
            points: (B, 3)
    return: interpolated_features: (B, dim)
    '''
    # Get grid resolution and batch size
    N = features.shape[1]
    B = points.shape[0]

    # Rescale points from (-1, 1) to (0, N-1)
    points = (points + 1) / 2 * (N - 1)
    # Create mask for points outside the grid, their values will be zero
    mask = torch.all((points > 0.)*(points < (N-1.)),dim=-1,keepdim=True) # (B, 1)
    # Points outside the grid are clamped to the grid boundaries (0,N-1)
    # Get voxel coordinates
    i_floor = torch.clamp(torch.floor(points[:, 0]), min=0., max=(N-1.)).long() # (B)
    j_floor = torch.clamp(torch.floor(points[:, 1]), min=0., max=(N-1.)).long()
    k_floor = torch.clamp(torch.floor(points[:, 2]), min=0., max=(N-1.)).long()

    i_ceil = torch.min(i_floor + 1, torch.tensor(N - 1, device=features.device))
    j_ceil = torch.min(j_floor + 1, torch.tensor(N - 1, device=features.device))
    k_ceil = torch.min(k_floor + 1, torch.tensor(N - 1, device=features.device))

    # Compute weights
    d_i = points[:, 0] - i_floor.float() # (B)
    d_j = points[:, 1] - j_floor.float()
    d_k = points[:, 2] - k_floor.float()

    w_000 = (1 - d_i) * (1 - d_j) * (1 - d_k) # (B)
    w_001 = (1 - d_i) * (1 - d_j) * d_k
    w_010 = (1 - d_i) * d_j * (1 - d_k)
    w_011 = (1 - d_i) * d_j * d_k
    w_100 = d_i * (1 - d_j) * (1 - d_k)
    w_101 = d_i * (1 - d_j) * d_k
    w_110 = d_i * d_j * (1 - d_k)
    w_111 = d_i * d_j * d_k

    # Gather features
    f_000 = torch.t(features[:, i_floor, j_floor, k_floor]).view(B, -1) 
    f_001 = torch.t(features[:, i_floor, j_floor, k_ceil]).view(B, -1)
    f_010 = torch.t(features[:, i_floor, j_ceil, k_floor]).view(B, -1)
    f_011 = torch.t(features[:, i_floor, j_ceil, k_ceil]).view(B, -1)
    f_100 = torch.t(features[:, i_ceil, j_floor, k_floor]).view(B, -1)
    f_101 = torch.t(features[:, i_ceil, j_floor, k_ceil]).view(B, -1)
    f_110 = torch.t(features[:, i_ceil, j_ceil, k_floor]).view(B, -1)
    f_111 = torch.t(features[:, i_ceil, j_ceil, k_ceil]).view(B, -1)

    # Compute interpolated features
    interpolated_features = (
        w_000[:, None] * f_000 +
        w_001[:, None] * f_001 +
        w_010[:, None] * f_010 +
        w_011[:, None] * f_011 +
        w_100[:, None] * f_100 +
        w_101[:, None] * f_101 +
        w_110[:, None] * f_110 +
        w_111[:, None] * f_111
    )
    masked_interpolated_features = interpolated_features * mask
    return masked_interpolated_features