"""
Sliding window transformer Model in Mirror contrastive loss based sliding window transformer.

Written by Jing Luo from Xi'an University of Technology, China.

luojing@xaut.edu.cn
"""
import numpy as np
import torch.nn.functional as F
from torch import Tensor
import random
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn as nn
from utils import trunc_normal_

seed_n = np.random.randint(500)

random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size: int = 40, conv_len: int = 25):
        """
        Initializes the PatchEmbedding layer.

        Args:
            emb_size (int): The size of embedding feature.
            conv_len (int): The length of the 1st convolution filter.
        """
        super().__init__()
        self.channel_num = 3
        self.in_channels = 1
        self.projection = nn.Sequential(
            # Swap the time dimension of the EEG with the channel dimension
            Rearrange('b e h w -> b e w h'),
            nn.Conv2d(self.in_channels, emb_size, (conv_len, 1), stride=(1, 1)),
            nn.Conv2d(emb_size, emb_size, (1, self.channel_num), stride=(1, 1)),
            nn.BatchNorm2d(emb_size, momentum=0.1, eps=1e-5),
            Rearrange('b e h w -> b (h w) e')
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.0):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class SlidingMultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 40, num_heads: int = 8, slide_len=8, slide_num=137, shift=False,
                 dropout: float = 0.0):
        super().__init__()
        """
        Initializes the SlidingMultiHeadAttention module.

        Args:
            emb_size (int): The size of embedding features.
            num_heads (int): The number of attention heads.
            slide_len (int): The length of each sliding window.
            slide_num (int): The number of sliding windows.
            shift (bool): Whether to use shifted windows.
            dropout (float): Dropout rate for attention.
        """
        self.slide_len = slide_len
        self.slide_num = slide_num
        self.shift = shift
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.softmax = nn.Softmax(dim=-1)

        # Relative position embedding
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * self.slide_len - 1), num_heads))
        coords = torch.zeros((self.slide_len, self.slide_len))
        coords[0, 0:] = torch.arange(self.slide_len)
        for i in range(1, self.slide_len):
            coords[i, :] = coords[i - 1, :] - 1
        relative_position_index = coords + self.slide_len - 1
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        # Temporal window or Sliding temporal window
        if self.shift:
            # Calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, self.slide_len * self.slide_num, 1))  # linear_len H W linear_len
            # Obtain three windows by slicing
            t_slices = (slice(0, -self.slide_len),
                        slice(-self.slide_len, -self.slide_len // 2),
                        slice(-self.slide_len // 2, None))
            # Index numbers to different windows
            cnt = 0
            for t in t_slices:
                img_mask[:, t, :] = cnt
                cnt += 1
            # Windows partition
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.squeeze(2)
            # Adjacent or not
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            # Set -100 if not adjacent
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def window_partition(self, x):
        b, h, c = x.shape
        x = x.view(b, h // self.slide_len, self.slide_len, c)
        windows = x.contiguous().view(-1, self.slide_len, c)
        return windows

    def window_reverse(self, windows):
        b = int(windows.shape[0] // self.slide_num)
        x = windows.view(b, self.slide_num, self.slide_len, -1)
        x = x.contiguous().view(b, self.slide_num * self.slide_len, -1)
        return x

    def multi_head_attention(self, x: Tensor, mask=None):
        b, N, C = x.shape
        # Attention scores
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        scaling = self.emb_size ** (1 / 2)
        energy = energy / scaling

        # Relative position embedding
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1).long()].view(
            self.slide_len, self.slide_len, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        energy = energy + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            energy = energy.view(b // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            energy = energy.view(-1, self.num_heads, N, N)
            att = F.softmax(energy, dim=-1)
        else:
            att = F.softmax(energy, dim=-1)

        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

    def forward(self, x: Tensor):
        if self.shift:
            x = torch.roll(x, shifts=(-self.slide_len // 2), dims=1)
            shift_window = self.window_partition(x)
            shift_window = self.multi_head_attention(shift_window, mask=self.attn_mask)
            shift_window = self.window_reverse(shift_window)
            x = torch.roll(shift_window, shifts=(self.slide_len // 2), dims=1)
        else:
            x = self.window_partition(x)
            x = self.multi_head_attention(x, mask=self.attn_mask)
            x = self.window_reverse(x)
        return x


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 40,
                 slide_num: int = 4,
                 drop_p: float = 0.0,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.0,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                ResidualAdd(nn.Sequential(
                    nn.LayerNorm(emb_size),
                    SlidingMultiHeadAttention(emb_size, slide_num=slide_num, shift=False, **kwargs),
                    nn.Dropout(drop_p)
                )),
                ResidualAdd(nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p)
                )),
                ResidualAdd(nn.Sequential(
                    nn.LayerNorm(emb_size),
                    SlidingMultiHeadAttention(emb_size, slide_num=slide_num, shift=True, **kwargs),
                    nn.Dropout(drop_p)
                )),
                ResidualAdd(nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p)
                ))
            ))
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 6, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, input_size: int = 1100, emb_size: int = 40, slide_len=275, n_classes: int = 2):
        super().__init__()
        self.poolk = 75
        self.pools = 15
        self.slide_len = slide_len
        self.linear_len = (input_size - self.poolk) // self.pools + 1
        self.pool = nn.AvgPool2d(kernel_size=(1, self.poolk), stride=(1, self.pools))
        self.drop = nn.Dropout(p=0.5)
        self.logS = nn.LogSoftmax(dim=1)
        self.soft = nn.Softmax(dim=1)
        self.gelu = nn.GELU()
        self.linear = nn.Linear(self.linear_len * emb_size, n_classes)
        self.linear1 = nn.Linear(self.linear_len * emb_size, emb_size)
        self.linear2 = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        x = torch.square(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.pool(x)
        x = torch.log(x)
        x = self.drop(x)
        x = rearrange(x, "b h n -> b (n h)")
        # Features applied in the contrastive learning
        contra_feature = F.normalize(x, p=2, dim=-1)

        x = self.linear1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.linear2(x)

        return contra_feature, x


class SWT(nn.Sequential):
    def __init__(self,
                 emb_size: int = 40,
                 eeg_size: int = 1120,
                 depth: int = 1,
                 slide_len: int = 8,
                 slide_num: int = 137,
                 n_classes: int = 2,
                 conv_len: int = 25,
                 **kwargs):
        super().__init__(
            PatchEmbedding(emb_size, conv_len),
            TransformerEncoder(depth, slide_num=slide_num, emb_size=emb_size, **kwargs),
            ClassificationHead(eeg_size - conv_len + 1, emb_size, slide_len, n_classes)
        )
