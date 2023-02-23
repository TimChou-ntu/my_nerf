import torch
from torch import nn

class Positional_Embedding(nn.Module):
    def __init__(self, n_freq, log=True):
        super(Positional_Embedding, self).__init__()
        self.n_freq = n_freq
        self.funcs = [torch.sin, torch.cos]
        if log:
            self.freq_bands = 2**torch.linspace(0, (self.n_freq-1), self.n_freq)
        else:
            self.freq_bands = torch.linspace(2**0, 2**(self.n_freq-1), self.n_freq)

    def forward(self, x):
        # not sure if this is [x] or []
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]
        
        return torch.cat(out, -1)

class NeRF(nn.Module):
    def __init__(self,
                D=8,
                W=256,
                input_channel_xyz=63,
                input_channel_dir=27,
                skips=[4]):
        super().__init__()
        self.D = D
        self.W = W
        self.input_channel_xyz = input_channel_xyz
        self.input_channel_dir = input_channel_dir
        self.skips = skips

        self.act = nn.functional.relu

        self.xyz_layers = nn.ModuleList(
            [nn.Linear(self.input_channel_xyz, self.W)] +
            [nn.Linear(self.W + self.input_channel_xyz, self.W) if (i+1) in self.skips \
            else nn.Linear(self.W, self.W) for i in range(D-1)]
        )
        self.xyz_encoding_final = nn.Linear(self.W, self.W)

        self.dir_layer = nn.Sequential(
            nn.Linear(self.W + self.input_channel_dir, W//2),
            nn.ReLU(True)
        )
        self.sigma = nn.Linear(self.W, 1)
        self.rgb_output = nn.Sequential(
            nn.Linear(self.W//2, 3),
            nn.Sigmoid())

    def forward(self, x, sigma_only=False):
        if not sigma_only:
            input_xyz, input_dir = torch.split(x, [self.input_channel_xyz, self.input_channel_dir], dim=-1)
        else:
            input_xyz = x
        
        xyz_ = input_xyz
        for i, layer in enumerate(self.xyz_layers):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz,xyz_], -1)
            xyz_ = self.act(layer(xyz_))
        
        sigma = self.sigma(xyz_)
        if sigma_only: return sigma

        xyz_encoding = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding, input_dir], -1)
        dir_encoding = self.dir_layer(dir_encoding_input)

        rgb = self.rgb_output(dir_encoding)

        out = torch.cat([rgb, sigma], -1)
        return out