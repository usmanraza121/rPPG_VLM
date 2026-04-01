import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# modified temporal cross attention
class TCA(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.k = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.v = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

        self.proj = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x, y):
        """
        x attends to y
        x, y: [B, C, T]
        """

        B, C, T = x.shape

        q = self.q(x)
        k = self.k(y)
        v = self.v(y)

        # [B, h, T, c]
        q = rearrange(q, 'b (h c) t -> b h t c', h=self.num_heads)
        k = rearrange(k, 'b (h c) t -> b h t c', h=self.num_heads)
        v = rearrange(v, 'b (h c) t -> b h t c', h=self.num_heads)

        # Normalize to preserve phase sensitivity
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Temporal attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)  # [B, h, T, c]
        out = rearrange(out, 'b h t c -> b (h c) t')

        return self.proj(out)

# =============================Temporal Enhancement Module==============================
class TemporalIEL(nn.Module):
    def __init__(self, dim, expansion=2):
        super().__init__()
        hidden = int(dim * expansion)

        self.project_in = nn.Conv1d(dim, hidden * 2, kernel_size=1)
        self.dwconv1 = nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden)
        self.dwconv2 = nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden)
        self.project_out = nn.Conv1d(hidden, dim, kernel_size=1)
        self.act = nn.Tanh()

    def forward(self, x):  # [B, C, T]
        x = self.project_in(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.act(self.dwconv1(x1)) + x1
        x2 = self.act(self.dwconv2(x2)) + x2
        return self.project_out(x1 * x2)

class TemporalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv1d(dim, dim, 1)
        self.kv = nn.Conv1d(dim, dim * 2, 1)
        self.project = nn.Conv1d(dim, dim, 1)

    def forward(self, x, y):  # x attends to y
        B, C, T = x.shape

        q = self.q(x)
        k, v = self.kv(y).chunk(2, dim=1)

        q = rearrange(q, 'b (h c) t -> b h c t', h=self.num_heads)
        k = rearrange(k, 'b (h c) t -> b h c t', h=self.num_heads)
        v = rearrange(v, 'b (h c) t -> b h c t', h=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b h c t -> b (h c) t')
        return self.project(out)

class TemporalLCA(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm_x = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)

        # self.attn = TemporalCrossAttention(dim, num_heads) # use either TCA or TemporalCrossAttention
        self.attn = TCA(dim, num_heads)
        self.iel = TemporalIEL(dim)

    def forward(self, x, y):
        # x, y: [B, T, C]
        x_ = self.norm_x(x).transpose(1, 2)
        y_ = self.norm_y(y).transpose(1, 2)

        x = x + self.attn(x_, y_).transpose(1, 2)
        x = x + self.iel(self.norm_x(x).transpose(1, 2)).transpose(1, 2)
        # x = x + self.iel(x.transpose(1, 2)).transpose(1, 2)
        return x


# ==============================Spatial Enhancement Module==============================
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        # Learnable weight parameters, initialized to all 1s
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        # Learnable bias parameters, initialized to all 0s
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        # Small constant for numerical stability
        self.eps = eps
        # Data format, supports "channels_last" and "channels_first"
        self.data_format = data_format
        # Check if the data format is valid, otherwise raise an exception
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        # Normalized shape, stored as a tuple
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        # If the data format is "channels_last"
        if self.data_format == "channels_last":
            # Directly call PyTorch's layer normalization function with the appropriate parameters
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # If the data format is "channels_first"
        elif self.data_format == "channels_first":
            # Calculate the mean along the channel dimension
            u = x.mean(1, keepdim=True)
            # Calculate the variance along the channel dimension
            s = (x - u).pow(2).mean(1, keepdim=True)
            # Perform normalization
            x = (x - u) / torch.sqrt(s + self.eps)
            # Apply learnable weight and bias parameters
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# Intensity Enhancement Layer
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        
        super(IEL, self).__init__()
       
        hidden_features = int(dim * ffn_expansion_factor)
       
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        
        self.Tanh = nn.Tanh()

    def forward(self, x):
        
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
       
        x = self.project_out(x)
        return x

# Cross Attention Block，
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        # 
        super(CAB, self).__init__()
        
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        
        b, c, h, w = x.shape
        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
       
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        #
        attn = nn.functional.softmax(attn, dim=-1)
        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        
        out = self.project_out(out)
        return out

# Lightweight Cross Attention，
class LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        #
        super(LCA, self).__init__()
        #
        self.gdfn = IEL(dim)
        # LayerNorm
        self.norm = LayerNorm(dim)
        # Cross Attention Block
        self.ffn = CAB(dim, num_heads, bias)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = self.gdfn(self.norm(x)) + x
        return x

if __name__ == "__main__":

    model = TemporalLCA(dim=64,num_heads=8)
    input_x = torch.randn(2,10,64)
    input_y = torch.randn(2,10,64)
    output = model(input_x, input_y)
    print('Input size:', input_x.size())
    print('Output size:', output.size())