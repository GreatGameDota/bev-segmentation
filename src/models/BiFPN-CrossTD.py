import torch
from torch import nn
import torch.nn.functional as F

import math
import timm
from effdet.efficientdet import BiFpn, ConvBnAct2d

# From: https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32).to(x.device)
        x_embed = not_mask.cumsum(2, dtype=torch.float32).to(x.device)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return x + pos

class Encoder(nn.Module):
  def __init__(self, in_f, out_f, out_bev):
    super(Encoder, self).__init__()
    self.out_f = out_f
    self.out_bev = out_bev

    self.proj_k = nn.Conv2d(in_f, out_f, kernel_size=1, bias=True)
    self.proj_v = nn.Conv2d(in_f, out_f, kernel_size=1, bias=True)

    self.pool = nn.AdaptiveAvgPool2d(1)
    # self.pe = PositionEmbeddingLearned(out_bev, num_pos_feats=in_f)
    self.pe = PositionEmbeddingSine(in_f // 2, normalize=True)
    self.mlp = nn.Linear(in_f, out_f)
  
  def forward(self, x):
    B, C, H, W = x.shape
    
    k = F.relu(self.proj_k(x))
    v = F.relu(self.proj_v(x))
    
    x_pool = self.pool(x)
    x_tile = torch.tile(x_pool, (1,self.out_bev,self.out_bev))
    
    mask = torch.zeros((B,self.out_bev,self.out_bev), dtype=torch.bool)
    cs = self.pe(x_tile, mask)
    
    # BxCxHxW -> H*WxBxC
    cs = cs.flatten(2).permute(2, 0, 1)
    q = F.relu(self.mlp(cs), inplace=True)
    q = q.permute(1, 2, 0).view(B, self.out_f, self.out_bev, self.out_bev)
    
    return [k, v, q]

# Based on: https://github.com/facebookresearch/detr/blob/main/models/transformer.py
class Decoder(nn.Module):
  def __init__(self, in_f, out_bev, nhead=8, dropout=0.1):
    super(Decoder, self).__init__()
    self.out_bev = out_bev
    self.in_f = in_f

    # self.self_attn = nn.MultiheadAttention(in_f, nhead, dropout=dropout)
    self.multihead_attn = nn.MultiheadAttention(in_f, nhead, dropout=dropout)

    self.norm1 = nn.LayerNorm(in_f)
    self.dropout1 = nn.Dropout(dropout)
    # self.norm2 = nn.LayerNorm(in_f)
    # self.dropout2 = nn.Dropout(dropout)

  def forward(self, x):
    k, v, q = x
    B, _, _, _ = k.shape
    # BxCxHxW -> H*WxBxC
    k = k.flatten(2).permute(2, 0, 1)
    v = v.flatten(2).permute(2, 0, 1)
    q = q.flatten(2).permute(2, 0, 1)

    res = self.multihead_attn(q, k, v)[0]
    q = q + self.dropout1(res)
    q = self.norm1(q)

    # res = self.multihead_attn(q, v, v)[0]
    # q = q + self.dropout2(res)
    # q = self.norm2(q)
    
    # Feed Forward
    # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    # tgt = tgt + self.dropout3(tgt2)
    # tgt = self.norm3(tgt)
    
    return q.permute(1, 2, 0).view(B, self.in_f, self.out_bev, self.out_bev)

class BiFPN_CrossTD(nn.Module):
  """ Image to BEV converter with BiFPN backbone to Cross Attention Transformer Decoder """
  def __init__(self, fpn_config, backbone_name, hidden_dim=512, out_f=16, out_sz=128, num_classes=10, return_features=False):
    super(BiFPN_CrossTD, self).__init__()
    self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True, out_indices=(2,3,4))
    feature_info = self.backbone.feature_info.get_dicts(keys=['num_chs', 'reduction'])
    self.fpn = BiFpn(fpn_config, feature_info)
    self.levels = fpn_config.num_levels

    out_f = out_sz // 8
    self.encoder = Encoder(in_f=fpn_config.fpn_channels, out_f=hidden_dim, out_bev=out_f)
    self.decoder = Decoder(in_f=hidden_dim, out_bev=out_f)

    self.blocks = nn.ModuleList()
    for i in range(3):
      self.blocks.append(nn.Sequential(
          nn.Conv2d(hidden_dim * fpn_config.num_levels if i == 0 else hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
          nn.BatchNorm2d(hidden_dim),
          nn.ReLU(inplace=True), # maybe try nn.SiLU
          nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
          nn.BatchNorm2d(hidden_dim),
          nn.ReLU(inplace=True)
      ))

    conv_args = dict(in_channels=hidden_dim, out_channels=num_classes, kernel_size=1, 
                     padding=fpn_config.pad_type, bias=True, norm_layer=None, act_layer=None)
    self.final_conv = ConvBnAct2d(**conv_args)
    self.return_features = return_features

  def freeze_backbone(self):
    for param in self.backbone.parameters():
      param.requires_grad = False

  def unfreeze_backbone(self):
    for param in self.backbone.parameters():
      param.requires_grad = True
  
  def features(self, x):
    x = self.backbone(x)
    x = self.fpn(x)

    xs = []
    for level in x:
      enc = self.encoder(level)
      xs.append(enc)

    feats = []
    for x in xs:
      x = self.decoder(x)
      feats.append(x)
    feats = torch.stack(feats)

    return feats

  def forward(self, x):
    x = self.backbone(x)
    x = self.fpn(x)

    xs = []
    for level in x:
      enc = self.encoder(level)
      xs.append(enc)

    feats = []
    for x in xs:
      x = self.decoder(x)
      feats.append(x)
    feats = torch.stack(feats)

    if self.return_features:
      return feats

    x = torch.tensor([]).to(feats[0].device)
    for feat in feats:
      x = torch.cat((x, feat),dim=1)

    for block in self.blocks:
      x = F.interpolate(x, scale_factor=2, mode="nearest")
      x = block(x)

    return self.final_conv(x)

def create_model(model_cfg, train_cfg):
    fpn_config = get_efficientdet_config(model_cfg['bifpn'])
    model = BiFPN_CrossTD(fpn_config, model_cfg['backbone'], hidden_dim=model_cfg['hidden_dim'], out_sz=train_cfg['output_sz'], num_classes=train_cfg['classes'])

    return model