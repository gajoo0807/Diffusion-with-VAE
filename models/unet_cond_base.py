import torch
from einops import einsum
import torch.nn as nn
from models.blocks import get_time_embedding
from models.blocks import DownBlock, MidBlock, UpBlockUnet
from utils.config_utils import *


class Unet(nn.Module):
    r"""
    Unet model comprising
    Down blocks, Midblocks and Uplocks
    更改Unet Block設計,加入distribution condition,並將class、text、image condition刪除
    """
    
    def __init__(self, im_channels, model_config):
        # diffusion model config, ldm params
        # 更改condition types: class -> distribution
        # 把class、text、image刪掉，加入distribution
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.t_emb_dim = model_config['time_emb_dim']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        self.attns = model_config['attn_down']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']
        self.conv_out_channels = model_config['conv_out_channels']
    

        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)
        ###################################
        
        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )
        
        self.up_sample = list(reversed(self.down_sample))
        self.downs = nn.ModuleList([])
        
        # Build the Downblocks
        for i in range(len(self.down_channels) - 1):
            # Cross Attention and Context Dim only needed if text condition is present
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i + 1], self.t_emb_dim,
                                        down_sample=self.down_sample[i],
                                        num_heads=self.num_heads,
                                        num_layers=self.num_down_layers,
                                        attn=self.attns[i], norm_channels=self.norm_channels))
        
        self.mids = nn.ModuleList([])
        # Build the Midblocks
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1], self.t_emb_dim,
                                      num_heads=self.num_heads,
                                      num_layers=self.num_mid_layers,
                                      norm_channels=self.norm_channels))
                
        self.ups = nn.ModuleList([])
        # Build the Upblocks
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(
                UpBlockUnet(self.down_channels[i] * 2, self.down_channels[i - 1] if i != 0 else self.conv_out_channels,
                            self.t_emb_dim, up_sample=self.down_sample[i],
                            num_heads=self.num_heads,
                            num_layers=self.num_up_layers,
                            norm_channels=self.norm_channels))
        
        self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
        self.conv_out = nn.Conv2d(self.conv_out_channels, im_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t, dist_input=None):
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        '''
        img cond:將img 與 x 合併,並經過conv_in_concat
        text cond: 將context_hidden_states輸入每一層down、mid、up layer中 -> dist embed
        class cond: 將class_embed加入t_emb
        '''
        # B x C1 x H x W
        # t_emb -> B x t_emb_dim
        out = self.conv_in(x)
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        # dist_input = torch.randn((x.shape(0), 10)) # 假設其為distribution input, shape = (B, 10)

        down_outs = []
        
        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out, t_emb, dist_input)
        # down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
        # out B x C4 x H/4 x W/4
        
        for mid in self.mids:
            out = mid(out, t_emb, dist_input)
        # out B x C3 x H/4 x W/4
        
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb, dist_input)
            # out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        # out B x C x H x W
        return out

# 計算模型參數數量和佔用空間
def get_model_size(model):
    num_params = sum(p.numel() for p in model.parameters())
    num_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return num_params, num_bytes


if __name__ == '__main__':
    # 配置 Unet 模型
    model_config = {
        'down_channels': [64, 128, 256, 512],
        'mid_channels': [512, 256, 128],
        'time_emb_dim': 128,
        'down_sample': [True, True, False],
        'num_down_layers': 2,
        'num_mid_layers': 2,
        'num_up_layers': 2,
        'attn_down': [False, False, True],
        'norm_channels': 32,
        'num_heads': 8,
        'conv_out_channels': 64,
    }


    # 初始化 Unet 模型
    im_channels = 1  # 假設輸入圖像有 3 個通道 (例如 RGB 圖像)
    unet = Unet(im_channels, model_config)

    # 計算模型參數數量和佔用空間
    num_params, num_bytes = get_model_size(unet)
    print(f"Model Parameters: {num_params}")
    print(f"Model Size: {num_bytes / (1024 ** 2):.2f} MB") # 88.26 MB
