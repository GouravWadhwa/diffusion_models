import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
import math

class Activation(nn.Module):
    def __init__(self, activation='relu', inplace=False, leaky_relu_slope=0.01):
        super().__init__()

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=inplace)
        elif activation == 'silu':
            self.activation = nn.SiLU(inplace=inplace)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(inplace=inplace, negative_slope=leaky_relu_slope)
        elif activation == 'none':
            self.activation = nn.Identity()
        else:
            NotImplementedError('[Activation] activation is not implemented:', activation)
    
    def forward(self, x):
        return self.activation(x)
    
class Normalization(nn.Module):
    def __init__(self, norm_type='group', norm_groups=8, channels=None):
        super().__init__()

        if norm_type == 'group':
            self.norm = nn.GroupNorm(norm_groups, channels)
        elif norm_type == 'batch' and channels != None:
            self.norm = nn.BatchNorm2d(channels)
        elif norm_type == 'none':
            self.norm = nn.Identity()
        else:
            NotImplementedError('[Normalization] normalization is not implemented:', norm_type)

    def forward(self, x):
        return self.norm(x)

"""
Convolution Block
(With the scale shift for time embedding.)
"""
class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, norm_type='none', norm_groups=8, activation='silu'):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = Normalization(norm_type=norm_type, norm_groups=norm_groups, channels=out_channels)

        self.activation = Activation(activation)

    
    def forward(self, x, scale_shift=None):
        x = self.norm(self.conv(x))

        if scale_shift:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.activation(x)
    
"""
Downsample
(reduces the spatial size by a factor)
"""
class Downsample(nn.Module):
    def __init__(self, input_channels, output_channels=None, factor=2.0):
        super().__init__()

        self.factor = factor
        self.output_channels = input_channels
        if output_channels:
            self.output_channels = output_channels

        self.conv = Conv2dBlock(
            in_channels=input_channels * (factor ** 2), out_channels=self.output_channels, kernel_size=1, padding=0, norm_type='none', activation='none'
        )

    def forward(self, x):
        x = einops.rearrange(x, 'b c (h hf) (w wf) -> b (c hf wf) h w', hf=self.factor, wf=self.factor)
        return self.conv(x)
    
"""
Upsample
(increases the spatial size by a factor)
"""
class Upsample(nn.Module):
    def __init__(self, input_channels, output_channels=None, factor=2.0):
        super().__init__()

        self.factor = factor
        self.output_channels = input_channels
        if output_channels is not None:
            self.output_channels = output_channels

        self.conv = Conv2dBlock(
            in_channels=input_channels, out_channels=self.output_channels * (factor ** 2), kernel_size=1, padding=0, norm_type='none', activation='none'
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = einops.rearrange(x, 'b (c hf wf) h w -> b c (h hf) (w wf)', hf=self.factor, wf=self.factor)

        return x

"""
ResNet Block
(Dense layer for time embedding)
"""
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, norm_type='none', norm_groups=8, activation='silu'):
        super().__init__()

        self.mlp = None
        if time_emb_dim:
            self.mlp = nn.Sequential(
                Activation(activation=activation),
                nn.Linear(time_emb_dim, 2 * out_channels)
            )

        self.block1 = Conv2dBlock(in_channels=in_channels, out_channels=out_channels, norm_type=norm_type, norm_groups=norm_groups, activation=activation)
        self.block2 = Conv2dBlock(in_channels=out_channels, out_channels=out_channels, norm_type=norm_type, norm_groups=norm_groups, activation='none')

        self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.activation = Activation(activation)


    def forward(self, x, time_emb=None):
        scale_shift = None
        if time_emb is not None and self.mlp is not None:
            time_emb = self.mlp(time_emb)
            time_emb = einops.rearrange(time_emb, 'b c -> b c 1 1')

            scale_shift = torch.chunk(time_emb, chunks=2, dim=1)

        emb = self.block2(self.block1(x, scale_shift))
        return self.activation(emb + self.shortcut(x))

"""
Attention Block
"""
class Attention(nn.Module):
    def __init__(self, channels, num_heads=8, channel_head=None, skip_connect=True, activation='silu', norm_type='none', norm_groups=8):
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.skip_connect = skip_connect

        if channel_head:
            self.channel_head = channel_head
        else:
            if self.channels % num_heads != 0:
                AssertionError(f"[Attention] number of channels:{channels} are not divisble by num_heads:{num_heads}")
            self.channel_head = self.channels // num_heads

        self.hidden_channels = self.num_heads * self.channel_head

        self.qkv_conv = Conv2dBlock(in_channels=self.channels, out_channels=3 * self.hidden_channels, kernel_size=1, padding=0, norm_type='none', activation=activation)
        self.out_conv = Conv2dBlock(in_channels=self.hidden_channels, out_channels=self.channels, kernel_size=1, padding=0, norm_type='none', activation='none')

        self.norm = Normalization(norm_type=norm_type, norm_groups=norm_groups, channels=self.channels)
    
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_conv(x)

        qkv = einops.rearrange(qkv, 'b (n c) h w -> b n c h w', c=self.channel_head * 3, n=self.num_heads)
        query, key, value = torch.chunk(qkv, 3, dim=2)

        attention = torch.einsum('bnchw, bncxy -> bnhwxy', query, key).contiguous() / torch.math.sqrt(self.channel_head)
        attention = einops.rearrange(attention, 'b n h w x y -> b n h w (x y)')
        attention = torch.softmax(attention, dim=-1)
        attention = einops.rearrange(attention, 'b n x y (h w) -> b n x y h w', h=h, w=w)

        output = torch.einsum('bnhwxy, bncxy -> bnchw', attention, value).contiguous()
        output = einops.rearrange(output, 'b n c h w -> b (n c) h w')

        output = self.out_conv(output)
        if self.skip_connect:
            output = output + x

        return self.norm(output)

"""
Feed Forward module
"""
class FeedForward(nn.Module):
    def __init__(self, input_channels, time_emb_channels=None, hidden_channels_factor=1.0, skip_connect=True, dropout=0.1, activation='silu', norm_type='none', norm_groups=8):
        super().__init__()

        self.input_channels = input_channels
        self.time_emb_channels = time_emb_channels
        self.hidden_channel_factor = hidden_channels_factor
        self.skip_connect = skip_connect

        self.hidden_channels = input_channels * hidden_channels_factor

        if time_emb_channels:
            self.to_scale_shift = nn.Sequential(
                Activation('silu'),
                nn.Linear(time_emb_channels, self.hidden_channels * 2)
            )

        self.proj_in = Conv2dBlock(input_channels, self.hidden_channels, kernel_size=1, padding=0, norm_type='none', activation=activation)
        self.proj_out = nn.Sequential(
            nn.Dropout(dropout),
            Conv2dBlock(self.hidden_channels, input_channels, kernel_size=1, padding=0, norm_type='none', activation='none')
        )
        
        self.norm = Normalization(norm_type=norm_type, norm_groups=norm_groups, channels=input_channels)

    def forward(self, inputs, time_emb=None):
        scale_shift = None
        if time_emb and self.to_scale_shift:
            time_emb = self.to_scale_shift(time_emb)
            time_emb = einops.rearrange(time_emb, 'b c -> b c 1 1')

            scale_shift = torch.chunk(time_emb, 2, dim=1)

        x = self.proj_in(inputs, scale_shift)
        x = self.proj_out(x)

        if self.skip_connect:
            x = x + inputs

        return self.norm(x)
    
"""
Transformer Block
"""
class TransformerBlock(nn.Module):
    def __init__(self, input_channels, time_emb_channels=None, num_heads=8, channel_head=None, ff_hidden_channels_factor=1.0, dropout=0.1, activation='silu', norm_type='none', norm_groups=8):
        super().__init__()

        self.attention_block = Attention(
            channels=input_channels, num_heads=num_heads, channel_head=channel_head, skip_connect=True, activation=activation, norm_type=norm_type, norm_groups=norm_groups
        )
        self.feed_forward = FeedForward(
            input_channels=input_channels, time_emb_channels=time_emb_channels, hidden_channels_factor=ff_hidden_channels_factor, skip_connect=True, dropout=dropout, activation=activation, norm_type=norm_type, norm_groups=norm_groups
        )
    
    def forward(self, x, time_emb=None):
        x = self.attention_block(x)
        x = self.feed_forward(x, time_emb)

        return x
    
"""
Learned Time Embedding
(returns a vector of size (b, dim + 1))
"""
class LearnedTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        assert dim % 2 == 0, "[Learned Time Embedding] dimension should be divisible by 2"

        self.weights = nn.Parameter(torch.randn(dim // 2))
    
    def forward(self, x):
        x = einops.rearrange(x, 'b -> b 1')
        frequency = torch.matmul(x, einops.rearrange(self.weights, 'd -> 1 d') * 2 * math.pi)

        fourier = torch.cat((frequency.sin(), frequency.cos()), dim=-1)
        fourier = torch.cat((x, fourier), dim=-1)

        return fourier
    
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000) / dim)
        )

        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        x = einops.rearrange(x, 'b -> b 1')
        frequency = torch.matmul(x, einops.rearrange(self.inv_freq, 'd -> 1 d'))

        fourier = torch.cat((frequency.sin(), frequency.cos()), dim=-1)
        fourier = torch.cat((x, fourier), dim=-1)

        return fourier
