import math
import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        freqs = torch.pow(10000, -torch.linspace(0, 1, n_embd//2))  # Shape: (1, n_embd//2)
        self.time_mlp = nn.Sequential(
                nn.Linear(n_embd, n_embd * 4),
                Swish(),
                nn.Linear(n_embd * 4, n_embd))
        self.register_buffer("freqs", freqs)

    def forward(self, time):  
        x = time[:, None] * self.freqs[None]
        # x = torch.tensor([time])[:, None].cuda() * self.freqs[None] 
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        return self.time_mlp(x)


class UpSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x, temb):
        return self.conv(self.up(x))
    

class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        # self.c2 = nn.Conv2d(in_ch, in_ch, 5, stride=2, padding=2)

    def forward(self, x, temb):
        x = self.c1(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch, torch_mha=False):
        super().__init__()
        self.torch_mha = torch_mha
        self.group_norm = nn.GroupNorm(32, in_ch)
        if torch_mha:
            self.mha = nn.MultiheadAttention(in_ch, 1, batch_first=True)
        else:
            self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
            self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
            self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)

        if self.torch_mha:
            h = h.view(-1, C, H * W).swapaxes(1, 2)
            h, _ = self.mha(h, h, h)
            h = h.swapaxes(2, 1).contiguous().view(-1, C, H, W)
        else:
            q = self.proj_q(h)
            k = self.proj_k(h)
            v = self.proj_v(h)

            q = q.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
            k = k.view(B, C, H * W)
            w = torch.bmm(q, k) * (int(C) ** (-0.5))
            assert list(w.shape) == [B, H * W, H * W]
            w = F.softmax(w, dim=-1)

            v = v.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
            h = torch.bmm(w, v)
            h = h.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        h = self.proj(h)
        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=True):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class Diffusion_UNet(nn.Module):
    def __init__(self, ch=64, ch_mult=[1, 2, 2, 2], num_res_blocks=2, dropout=0.15):
        super().__init__()
        tdim = ch
        self.time_embedding = TimeEmbedding(tdim)
        self.head = nn.Conv2d(2, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=False))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 1, 3, stride=1, padding=1)
        )
 
    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h
    

if __name__ == '__main__':
    model = Diffusion_UNet()
    summary(model, input_size=(1, 2, 128, 128))