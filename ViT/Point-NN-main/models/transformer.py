import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, Mlp

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        # every head_dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        # 一个变成3个 一个token -- > qkv
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, temp=1.0):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # q's dim is (b,heads,n,dim) , k's dim is (b,heads,dim,n)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # except matrix,there are indexs
        # softmax in last dimension, (b,heads,n,n)
        attn = (attn / temp).softmax(dim=-1)
        attn = self.attn_drop(attn)
        # attn = b,heads,n,n  v = b,heads,n,dim ---- > b,heads,n,dim
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=768,
                 bottleneck=64,
                 dropout=0.1,
                 init_option="lora",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in",):
        super().__init__()
        self.n_embd = d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck
        #_before
        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)
        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * self.scale
        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)
        if add_residual:
            output = up + residual
        else:
            output = up
        return output

class Block(nn.Module):

    def __init__(self, dim=768, num_heads=12, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.adaptmlp = Adapter(dropout=0.1, d_model=dim)

    def forward(self, x, temp=1.0):
        # b,n,c
        x = x + self.drop_path(self.attn(self.norm1(x), temp=temp))
        residual = x
        # mlp and adapt_mlp
        # adapt_x = self.adaptmlp(x)
        x = self.drop_path(self.mlp(self.norm2(x)))
        # parallel
        x = x + residual
        return x
    
    def calculate_emo_loss(self, patch_embeds):
        """
        计算 Patch Tokens 的 EMD 损失
        输入: patch_embeds [B, N, C] (N为Patch数量)
        输出: EMO 损失标量
        """
        B, N, C = patch_embeds.shape
        
        # 1. 构造目标分布（均匀分布或其他先验）
        target_dist = torch.ones(B, N, device=patch_embeds.device) / N  # [B, N]
        
        # 2. 计算成本矩阵（Patch 间余弦距离）
        patch_norm = patch_embeds / torch.norm(patch_embeds, dim=-1, keepdim=True)
        cost_matrix = torch.bmm(patch_norm, patch_norm.transpose(1, 2))  # [B, N, N]
        
        # 3. 用 Sinkhorn 算法近似 EMD
        emd_loss = self.sinkhorn_emd(cost_matrix, target_dist)
        return emd_loss.mean()  # 批次平均

    def sinkhorn_emd(self, cost_matrix, target_dist, epsilon=0.1, n_iters=5):
        """
        Sinkhorn 迭代求解 EMD
        输入: cost_matrix [B, N, N], target_dist [B, N]
        输出: EMD 损失 [B]
        """
        log_target = torch.log(target_dist + 1e-8)
        log_transport = -cost_matrix / epsilon  # 初始化为成本矩阵的负值
        
        for _ in range(n_iters):
            # 行归一化（保证每行是一个分布）
            log_transport = log_transport - torch.logsumexp(log_transport, dim=2, keepdim=True)
            # 列归一化（逼近目标分布）
            log_transport = log_transport - torch.logsumexp(log_transport, dim=1, keepdim=True) + log_target.unsqueeze(1)
        
        # 计算最终传输计划
        transport_plan = torch.exp(log_transport)
        emd = torch.sum(transport_plan * cost_matrix, dim=(1, 2))
        return emd