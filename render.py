import torch
from torch import searchsorted

def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.

    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero

    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[:,:1]),cdf], -1) # (N_rays, N_samples_+1)

    if det:
        u = torch.linspace(0,1,N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)

    u = u.contiguous()
    inds = searchsorted(cdf, u, side='right')
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)
    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples

def render_rays(models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                test_time=False
                ):

    def inference(model, embedding_xyz, xyz_, dir_, dir_embedded, z_vals, weights_only=False):
        N_samples_ = xyz_.shape[1]
        xyz_ = xyz_.view(-1,3)
        if not weights_only:
            dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
        
        B = xyz_.shape[0]
        out_chunks = []
        for i in range(0, B, chunk):
            xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
            if not weights_only:
                xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded[i:i+chunk]], 1)

            else:
                xyzdir_embedded = xyz_embedded
            out_chunks += [model(xyzdir_embedded, sigma_only=weights_only)]

        out = torch.cat(out_chunks, 0)

        # N_rays = rays.shape[0]
        if weights_only:
            sigmas = out.view(N_rays, N_samples_)
        else:
            rgbsigma = out.view(N_rays, N_samples_, 4)
            rgbs = rgbsigma[...,:3]
            sigmas = rgbsigma[...,3]

        deltas = z_vals[:, 1:] - z_vals[:,:-1] # (N_rays, N_samples-1)
        delta_inf = 1e10*torch.ones_like(deltas[:,:1])
        deltas = torch.cat([deltas, delta_inf], -1) # (N_rays, N_samples_)

        deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)
        noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

        alphas = 1 - torch.exp(-deltas*torch.relu(sigmas+noise))
        alphas_shifted = torch.cat([torch.ones_like(alphas[:,:1]), 1-alphas+1e-10], -1) #(N_rays, N_samples_+1)
        weights = alphas * torch.cumprod(alphas_shifted, -1)[:,:-1] #cumulative product, ex (1,2,3,4) -> (1,2,6,24)
        weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays, equals to 1-(1-a1)(1-a2)...(1-an)

        if weights_only: return weights

        rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
        depth_final = torch.sum(weights*z_vals, -1) #(N_rays)

        if white_back:
            rgb_final = rgb_final + 1 - weights_sum.unsqueeze(-1)
        
        return rgb_final, depth_final, weights

    model_coarse = models[0]
    embedding_xyz = embeddings[0]
    embedding_dir = embeddings[1]

    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:,0:3], rays[:,3:6] # (N_rays, 3)
    near, far = rays[:,6:7], rays[:,7:8]

    dir_embedded = embedding_dir(rays_d) # (N_rays, embed_dir_channels)

    z_steps = torch.linspace(0,1,N_samples,device=rays.device) # (N_samples)
    if not use_disp:            # using linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else:                       # using linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)

    if perturb > 0:
        z_vals_mid = 0.5 *(z_vals[:,:-1] + z_vals[:,1:]) # (N_rays, N_samples-1) interval mid points
        upper = torch.cat([z_vals_mid, z_vals[:,-1:]], -1)
        lower = torch.cat([z_vals[:,:1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand(z_vals.shape, device = rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse_sampled = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2) #(N_rays, N_samples, 3)

    if test_time:
        weights_coarse = inference(model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d, dir_embedded, z_vals, weights_only=True)
        result = {'opacity_coarse': weights_coarse.sum(1)}
    else:
        rgb_coarse, depth_coarse, weights_coarse = \
            inference(model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d, dir_embedded, z_vals, weights_only=False)
        result = {
            'rgb_coarse':rgb_coarse,
            'depth_coarse':depth_coarse,
            'opacity_coarse':weights_coarse.sum(1)
        }
    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5*(z_vals[:,:-1]+z_vals[:,1:]) # (N_rays, N_samples-1) middle point
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb==0)).detach()
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        xyz_fine_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
                           # (N_rays, N_samples+N_importance, 3)

        model_fine = models[1]
        rgb_fine, depth_fine, weights_fine = \
            inference(model_fine, embedding_xyz, xyz_fine_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False)

        result['rgb_fine'] = rgb_fine
        result['depth_fine'] = depth_fine
        result['opacity_fine'] = weights_fine.sum(1)

    return result