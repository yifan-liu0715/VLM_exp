import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List
from einops import rearrange
from .wan_video_camera_controller import SimpleAdapter
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
from torch.nn.attention.flex_attention import create_block_mask, flex_attention,BlockMask
# mode="max-autotune" does not work here on H100 under FSDP, for two reasons:
#   - it turns on cudagraphs, and FSDP frees/reallocates the flat parameter
#     between passes, which trips "graph recording observed an input tensor
#     deallocate during graph recording that did not occur during replay";
#   - its larger candidate block sizes exceed the 232448-byte shared memory
#     limit at head_dim 128 ("No valid triton configs ... Required: 233472").
# The default mode keeps the compile without either problem. Set
# DIFFSYNTH_FLEX_COMPILE_MODE to override, or "eager" to skip compiling.
_flex_compile_mode = os.environ.get("DIFFSYNTH_FLEX_COMPILE_MODE", "default")
if _flex_compile_mode != "eager":
    flex_attention = torch.compile(
        flex_attention, dynamic=False, mode=_flex_compile_mode)


######## wan animate adapter components ######
MEMORY_LAYOUT = {
    "flash": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
    ),
    "torch": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
}


def attention(
    q,
    k,
    v,
    mode="torch",
    drop_rate=0,
    attn_mask=None,
    causal=False,
    max_seqlen_q=None,
    batch_size=1,
):
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]

    if mode == "torch":
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal)

    x = post_attn_layout(x)
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out


class CausalConv1d(nn.Module):

    def __init__(self, chan_in, chan_out, kernel_size=3, stride=1, dilation=1, pad_mode="replicate", **kwargs):
        super().__init__()

        self.pad_mode = pad_mode
        padding = (kernel_size - 1, 0)  # T
        self.time_causal_padding = padding

        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)



class FaceEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_heads=int, dtype=None, device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

        self.num_heads = num_heads
        self.conv1_local = CausalConv1d(in_dim, 1024 * num_heads, 3, stride=1)
        self.norm1 = nn.LayerNorm(hidden_dim // 8, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.act = nn.SiLU()
        self.conv2 = CausalConv1d(1024, 1024, 3, stride=2)
        self.conv3 = CausalConv1d(1024, 1024, 3, stride=2)

        self.out_proj = nn.Linear(1024, hidden_dim)
        self.norm1 = nn.LayerNorm(1024, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.norm2 = nn.LayerNorm(1024, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.norm3 = nn.LayerNorm(1024, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def forward(self, x):
        
        x = rearrange(x, "b t c -> b c t")
        b, c, t = x.shape

        x = self.conv1_local(x)
        x = rearrange(x, "b (n c) t -> (b n) t c", n=self.num_heads)
        
        x = self.norm1(x)
        x = self.act(x)
        x = rearrange(x, "b t c -> b c t")
        x = self.conv2(x)
        x = rearrange(x, "b c t -> b t c")
        x = self.norm2(x)
        x = self.act(x)
        x = rearrange(x, "b t c -> b c t")
        x = self.conv3(x)
        x = rearrange(x, "b c t -> b t c")
        x = self.norm3(x)
        x = self.act(x)
        x = self.out_proj(x)
        x = rearrange(x, "(b n) t c -> b t n c", b=b)
        padding = self.padding_tokens.repeat(b, x.shape[1], 1, 1)
        x = torch.cat([x, padding], dim=-2)
        x_local = x.clone()

        return x_local



class RMSNorm_(nn.Module):
    def __init__(
        self,
        dim: int,
        elementwise_affine=True,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


def get_norm_layer(norm_layer):
    """
    Get the normalization layer.

    Args:
        norm_layer (str): The type of normalization layer.

    Returns:
        norm_layer (nn.Module): The normalization layer.
    """
    if norm_layer == "layer":
        return nn.LayerNorm
    elif norm_layer == "rms":
        return RMSNorm_
    else:
        raise NotImplementedError(f"Norm layer {norm_layer} is not implemented")


class FaceAdapter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads_num: int,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        num_adapter_layers: int = 1,
        dtype=None,
        device=None,
    ):

        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.hidden_size = hidden_dim
        self.heads_num = heads_num
        self.fuser_blocks = nn.ModuleList(
            [
                FaceBlock(
                    self.hidden_size,
                    self.heads_num,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    **factory_kwargs,
                )
                for _ in range(num_adapter_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        motion_embed: torch.Tensor,
        idx: int,
        freqs_cis_q: Tuple[torch.Tensor, torch.Tensor] = None,
        freqs_cis_k: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:

        return self.fuser_blocks[idx](x, motion_embed, freqs_cis_q, freqs_cis_k)



class FaceBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        self.scale = qk_scale or head_dim**-0.5
       
        self.linear1_kv = nn.Linear(hidden_size, hidden_size * 2, **factory_kwargs)
        self.linear1_q = nn.Linear(hidden_size, hidden_size, **factory_kwargs)

        self.linear2 = nn.Linear(hidden_size, hidden_size, **factory_kwargs)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )

        self.pre_norm_feat = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.pre_norm_motion = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        motion_vec: torch.Tensor,
        motion_mask: Optional[torch.Tensor] = None,
        use_context_parallel=False,
    ) -> torch.Tensor:
        
        B, T, N, C = motion_vec.shape
        T_comp = T

        x_motion = self.pre_norm_motion(motion_vec)
        x_feat = self.pre_norm_feat(x)

        kv = self.linear1_kv(x_motion)
        q = self.linear1_q(x_feat)

        k, v = rearrange(kv, "B L N (K H D) -> K B L N H D", K=2, H=self.heads_num)
        q = rearrange(q, "B S (H D) -> B S H D", H=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        k = rearrange(k, "B L N H D -> (B L) H N D")  
        v = rearrange(v, "B L N H D -> (B L) H N D") 

        q = rearrange(q, "B (L S) H D -> (B L) H S D", L=T_comp)  
        # Compute attention.
        attn = F.scaled_dot_product_attention(q, k, v)

        attn = rearrange(attn, "(B L) H S D -> B (L S) (H D)", L=T_comp)

        output = self.linear2(attn)

        if motion_mask is not None:
            output = output * rearrange(motion_mask, "B T H W -> B (T H W)").unsqueeze(-1)

        return output



def custom_qr(input_tensor):
    original_dtype = input_tensor.dtype
    if original_dtype == torch.bfloat16:
        q, r = torch.linalg.qr(input_tensor.to(torch.float32))
        return q.to(original_dtype), r.to(original_dtype)
    return torch.linalg.qr(input_tensor)

def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
	return F.leaky_relu(input + bias, negative_slope) * scale


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
	_, minor, in_h, in_w = input.shape
	kernel_h, kernel_w = kernel.shape

	out = input.view(-1, minor, in_h, 1, in_w, 1)
	out = F.pad(out, [0, up_x - 1, 0, 0, 0, up_y - 1, 0, 0])
	out = out.view(-1, minor, in_h * up_y, in_w * up_x)

	out = F.pad(out, [max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
	out = out[:, :, max(-pad_y0, 0): out.shape[2] - max(-pad_y1, 0),
		  max(-pad_x0, 0): out.shape[3] - max(-pad_x1, 0), ]

	out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
	w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
	out = F.conv2d(out, w)
	out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
					  in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1, )
	return out[:, :, ::down_y, ::down_x]


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
	return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])


def make_kernel(k):
	k = torch.tensor(k, dtype=torch.float32)
	if k.ndim == 1:
		k = k[None, :] * k[:, None]
	k /= k.sum()
	return k


class FusedLeakyReLU(nn.Module):
	def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
		super().__init__()
		self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
		self.negative_slope = negative_slope
		self.scale = scale

	def forward(self, input):
		out = fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)
		return out


class Blur(nn.Module):
	def __init__(self, kernel, pad, upsample_factor=1):
		super().__init__()

		kernel = make_kernel(kernel)

		if upsample_factor > 1:
			kernel = kernel * (upsample_factor ** 2)

		self.kernel = torch.nn.Parameter(kernel)

		self.pad = pad

	def forward(self, input):
		return upfirdn2d(input, self.kernel, pad=self.pad)


class ScaledLeakyReLU(nn.Module):
	def __init__(self, negative_slope=0.2):
		super().__init__()

		self.negative_slope = negative_slope

	def forward(self, input):
		return F.leaky_relu(input, negative_slope=self.negative_slope)


class EqualConv2d(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
		super().__init__()

		self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
		self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

		self.stride = stride
		self.padding = padding

		if bias:
			self.bias = nn.Parameter(torch.zeros(out_channel))
		else:
			self.bias = None

	def forward(self, input):

		return F.conv2d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)

	def __repr__(self):
		return (
			f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
			f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
		)


class EqualLinear(nn.Module):
	def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
		super().__init__()

		self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

		if bias:
			self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
		else:
			self.bias = None

		self.activation = activation

		self.scale = (1 / math.sqrt(in_dim)) * lr_mul
		self.lr_mul = lr_mul

	def forward(self, input):

		if self.activation:
			out = F.linear(input, self.weight * self.scale)
			out = fused_leaky_relu(out, self.bias * self.lr_mul)
		else:
			out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

		return out

	def __repr__(self):
		return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})')


class ConvLayer(nn.Sequential):
	def __init__(
			self,
			in_channel,
			out_channel,
			kernel_size,
			downsample=False,
			blur_kernel=[1, 3, 3, 1],
			bias=True,
			activate=True,
	):
		layers = []

		if downsample:
			factor = 2
			p = (len(blur_kernel) - factor) + (kernel_size - 1)
			pad0 = (p + 1) // 2
			pad1 = p // 2

			layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

			stride = 2
			self.padding = 0

		else:
			stride = 1
			self.padding = kernel_size // 2

		layers.append(EqualConv2d(in_channel, out_channel, kernel_size, padding=self.padding, stride=stride,
								  bias=bias and not activate))

		if activate:
			if bias:
				layers.append(FusedLeakyReLU(out_channel))
			else:
				layers.append(ScaledLeakyReLU(0.2))

		super().__init__(*layers)


class ResBlock(nn.Module):
	def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
		super().__init__()

		self.conv1 = ConvLayer(in_channel, in_channel, 3)
		self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

		self.skip = ConvLayer(in_channel, out_channel, 1, downsample=True, activate=False, bias=False)

	def forward(self, input):
		out = self.conv1(input)
		out = self.conv2(out)

		skip = self.skip(input)
		out = (out + skip) / math.sqrt(2)

		return out


class EncoderApp(nn.Module):
	def __init__(self, size, w_dim=512):
		super(EncoderApp, self).__init__()

		channels = {
			4: 512,
			8: 512,
			16: 512,
			32: 512,
			64: 256,
			128: 128,
			256: 64,
			512: 32,
			1024: 16
		}

		self.w_dim = w_dim
		log_size = int(math.log(size, 2))

		self.convs = nn.ModuleList()
		self.convs.append(ConvLayer(3, channels[size], 1))

		in_channel = channels[size]
		for i in range(log_size, 2, -1):
			out_channel = channels[2 ** (i - 1)]
			self.convs.append(ResBlock(in_channel, out_channel))
			in_channel = out_channel

		self.convs.append(EqualConv2d(in_channel, self.w_dim, 4, padding=0, bias=False))

	def forward(self, x):

		res = []
		h = x
		for conv in self.convs:
			h = conv(h)
			res.append(h)

		return res[-1].squeeze(-1).squeeze(-1), res[::-1][2:]


class Encoder(nn.Module):
	def __init__(self, size, dim=512, dim_motion=20):
		super(Encoder, self).__init__()

		# appearance netmork
		self.net_app = EncoderApp(size, dim)

		# motion network
		fc = [EqualLinear(dim, dim)]
		for i in range(3):
			fc.append(EqualLinear(dim, dim))

		fc.append(EqualLinear(dim, dim_motion))
		self.fc = nn.Sequential(*fc)

	def enc_app(self, x):
		h_source = self.net_app(x)
		return h_source

	def enc_motion(self, x):
		h, _ = self.net_app(x)
		h_motion = self.fc(h)
		return h_motion


class Direction(nn.Module):
    def __init__(self, motion_dim):
        super(Direction, self).__init__()
        self.weight = nn.Parameter(torch.randn(512, motion_dim))

    def forward(self, input):

        weight = self.weight + 1e-8
        Q, R = custom_qr(weight)
        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)
            return out


class Synthesis(nn.Module):
    def __init__(self, motion_dim):
        super(Synthesis, self).__init__()
        self.direction = Direction(motion_dim)


class Generator(nn.Module):
    def __init__(self, size, style_dim=512, motion_dim=20):
        super().__init__()

        self.enc = Encoder(size, style_dim, motion_dim)
        self.dec = Synthesis(motion_dim)

    def get_motion(self, img):
        #motion_feat = self.enc.enc_motion(img)
        motion_feat = torch.utils.checkpoint.checkpoint((self.enc.enc_motion), img, use_reentrant=True)
        motion = self.dec.direction(motion_feat)
        return motion


###### diffsion model components ######
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)
    
    def forward(self, x, freqs, block_mask, kv_cache=None, current_start=0,seg_idx=None,frame_seqlen=1536):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x))
            k = self.norm_k(self.k(x))
            v = self.v(x)
            return q, k, v
        q, k, v = qkv_fn(x)
        
        if kv_cache is None:
            # teacher forcing layout for i2v:
            # [ref_tokens, clean_content_tokens, noisy_content_tokens]
            single_seq = freqs.shape[0]
            content_seq = max(single_seq - frame_seqlen, 0)
            is_tf = (content_seq > 0) and (s == frame_seqlen + content_seq * 2)
            if is_tf:
                q_ref = q[:, :frame_seqlen]
                q_clean = q[:, frame_seqlen:frame_seqlen + content_seq]
                q_noisy = q[:, frame_seqlen + content_seq:]

                k_ref = k[:, :frame_seqlen]
                k_clean = k[:, frame_seqlen:frame_seqlen + content_seq]
                k_noisy = k[:, frame_seqlen + content_seq:]

                freqs_ref = freqs[:frame_seqlen]
                freqs_content = freqs[frame_seqlen:]

                roped_query = torch.cat([
                    rope_apply(q_ref, freqs_ref, self.num_heads),
                    rope_apply(q_clean, freqs_content, self.num_heads),
                    rope_apply(q_noisy, freqs_content, self.num_heads),
                ], dim=1)
                roped_key = torch.cat([
                    rope_apply(k_ref, freqs_ref, self.num_heads),
                    rope_apply(k_clean, freqs_content, self.num_heads),
                    rope_apply(k_noisy, freqs_content, self.num_heads),
                ], dim=1)
            else:
                roped_query = rope_apply(q, freqs, self.num_heads)
                roped_key = rope_apply(k, freqs, self.num_heads)

            roped_query = roped_query.reshape(b, s, n, d)
            roped_key = roped_key.reshape(b, s, n, d)
            v = v.reshape(b, s, n, d)
            # pad zero to meet block size requirement
            padded_length = math.ceil(roped_query.shape[1] / 128) * 128 - roped_query.shape[1]
            padded_roped_query = torch.cat(
                [roped_query,
                 torch.zeros([roped_query.shape[0], padded_length, roped_query.shape[2], roped_query.shape[3]],
                             device=roped_query.device, dtype=v.dtype)],
                dim=1
            )
            padded_roped_key = torch.cat(
                [roped_key, torch.zeros([roped_key.shape[0], padded_length, roped_key.shape[2], roped_key.shape[3]],
                                        device=roped_key.device, dtype=roped_key.dtype)],
                dim=1
            )
            padded_v = torch.cat(
                [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                device=v.device, dtype=v.dtype)],
                dim=1
            )

            # use flex attention
            x = flex_attention(
                query=padded_roped_query.transpose(2, 1),
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask
            )
            if padded_length > 0:
                x = x[:, :, :-padded_length]
            x = x.transpose(2, 1)

        else:            
            roped_query = rope_apply(q, freqs, self.num_heads)
            roped_key = rope_apply(k, freqs, self.num_heads)
            
            kv_cache["k"][:, current_start:current_start+roped_key.shape[1]] = roped_key
            kv_cache["v"][:, current_start:current_start+roped_key.shape[1]] = v
            
            x_cond = self.attn(q=roped_query[:,:seg_idx[1]],
                    k=kv_cache["k"][:, current_start:current_start+seg_idx[1]],
                    v=kv_cache["v"][:, current_start:current_start+seg_idx[1]],
                    num_heads=self.num_heads)
            
            x_noise = self.attn(q=roped_query[:,seg_idx[1]:],
                    k=kv_cache["k"][:, :current_start+roped_key.shape[1]],
                    v=kv_cache["v"][:, :current_start+roped_key.shape[1]],
                    num_heads=self.num_heads)
            
            x = torch.cat([x_cond, x_noise],dim=1)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, cond_t_mod, freqs,
                block_mask=None,
                kv_cache=None,
                frame_seqlen=1536,
                current_start=0,
                ):
        
        ref_end = frame_seqlen
        seg_idx = [0, ref_end, x.size(1)]
        
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        
        cond_shift_msa, cond_scale_msa, cond_gate_msa, cond_shift_mlp, cond_scale_mlp, cond_gate_mlp = (
            self.modulation.to(dtype=cond_t_mod.dtype, device=cond_t_mod.device) + cond_t_mod).chunk(6, dim=chunk_dim)
        
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            shift_msa.squeeze(1), scale_msa.squeeze(1), gate_msa.squeeze(1),
            shift_mlp.squeeze(1), scale_mlp.squeeze(1), gate_mlp.squeeze(1),
        )
        cond_shift_msa, cond_scale_msa, cond_gate_msa, cond_shift_mlp, cond_scale_mlp, cond_gate_mlp = (
            cond_shift_msa.squeeze(1), cond_scale_msa.squeeze(1), cond_gate_msa.squeeze(1),
            cond_shift_mlp.squeeze(1), cond_scale_mlp.squeeze(1), cond_gate_mlp.squeeze(1),
        )
        
        ref_orig, content_orig = x.split([seg_idx[1], seg_idx[2] - seg_idx[1]], dim=1)
        ref = modulate(self.norm1(ref_orig), cond_shift_msa, cond_scale_msa)
        content = modulate(self.norm1(content_orig), shift_msa, scale_msa)
        
        input_x = torch.cat([ref, content], dim=1)
        x = self.self_attn(input_x, freqs, block_mask, kv_cache=kv_cache, current_start=current_start,seg_idx=seg_idx,frame_seqlen=frame_seqlen)
        ref, content = x.split([seg_idx[1], seg_idx[2] - seg_idx[1]], dim=1)
        
        ref_x = self.gate(ref_orig, cond_gate_msa, ref)
        content_x = self.gate(content_orig, gate_msa, content)
        x_res = torch.cat([ref_x, content_x], dim=1)
        
        x = torch.cat([self.norm3(ref_x), self.norm3(content_x)], dim=1)
        x = x_res + self.cross_attn(x, context)
        
        ref_x, content_x = x.split([seg_idx[1], seg_idx[2] - seg_idx[1]], dim=1)
        ref = modulate(self.norm2(ref_x), cond_shift_mlp, cond_scale_mlp)
        content = modulate(self.norm2(content_x), shift_mlp, scale_mlp)
        ref, content = self.ffn(ref), self.ffn(content)
        ref_x = self.gate(ref_x, cond_gate_mlp, ref)
        content_x = self.gate(content_x, gate_mlp, content)
        
        x = torch.cat([ref_x, content_x], dim=1)
        
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
    
    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        elif len(t_mod.shape) == 2:

            if t_mod.shape[0] == x.shape[0]: # uniform t_mod for all tokens in the batch
                import os
                if os.environ.get("DEBUG", 0) == 1:
                    print(f"Debug: t for head shape {t_mod.shape} matches batch size {x.shape[0]}, applying batch-level modulation.")
                    
                shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
                x = self.head(self.norm(x) * (1 + scale.squeeze(1)) + shift.squeeze(1))
            else:
                import os
                if os.environ.get("DEBUG", 0) == 1:
                    print(f"Debug: t for head shape {t_mod.shape} does not match batch size {x.shape[0]}, applying token-level modulation.")
                
                # (total_seq, dim): token-level，每个 token 对应自己的 t for diffusion forcing
                combined = self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(1)  # modulation:1 2 5120 ，t_mod为 14625 5120
                shift, scale = combined.chunk(2, dim=1)
                shift = shift.squeeze(1)                    # (S, dim)
                scale = scale.squeeze(1)                    # (S, dim)
                x = self.head(self.norm(x) * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0))
        return x


class CausalWanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None
            
        ####
        self.block_mask_cache = {}
        self.num_frame_per_block = 3
        self.local_attn_size = 6
        ####
        
        #### adapter related modules ####
        self.pose_patch_embedding = torch.nn.Conv3d(16, 5120, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.motion_encoder = Generator(size=512, style_dim=512, motion_dim=20)
        self.face_adapter = FaceAdapter(heads_num=40, hidden_dim=5120, num_adapter_layers=40 // 5)
        self.face_encoder = FaceEncoder(in_dim=512, hidden_dim=5120, num_heads=4)
        ####

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        b, c, f, h, w = x.shape
        x = self.patch_embedding(x)
        # Calculate grid size after patching
        f_patch = f // self.patch_size[0]
        h_patch = h // self.patch_size[1]
        w_patch = w // self.patch_size[2]
        
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x, (f_patch, h_patch, w_patch)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )
        
    def _forward_train(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                clean_x: Optional[torch.Tensor] = None,
                aug_t: Optional[torch.Tensor] = None,
                pose_latents: Optional[torch.Tensor] = None,
                face_pixel_values: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                start_frame: int = 0,
                **kwargs,):
        
        context = self.text_embedding(context)
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        clip_embdding = self.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)
        frame_seqlen = h * w  # 提前算好，后面 block mask 也用得到
        cond_lens = 1 # hard code reference 1 and motion frames 0
        def recompute_freqs(f, h, w, cond_lens):
            seq_end = f * h * w          # 所有帧的 token 数 (f 帧)
            ref_seq_len = cond_lens * h * w          # 参考帧的 token 数 (1 帧)
            content_seq_len = seq_end - ref_seq_len  # 内容帧的 token 数 (f-1 帧)
            freqs_ref = torch.cat([
                            self.freqs[0][30:30+cond_lens].view(cond_lens, 1, 1, -1).expand(cond_lens, h, w, -1),  # temporal
                            self.freqs[1][:h].view(1, h, 1, -1).expand(cond_lens, h, w, -1),  # height
                            self.freqs[2][:w].view(1, 1, w, -1).expand(cond_lens, h, w, -1)   # width
                        ], dim=-1).reshape(ref_seq_len, 1, -1).to(x.device)
            freqs_content = torch.cat([
                            self.freqs[0][start_frame:start_frame + f - cond_lens].view(f-cond_lens, 1, 1, -1).expand(f-cond_lens, h, w, -1),  # temporal
                            self.freqs[1][:h].view(1, h, 1, -1).expand(f-cond_lens, h, w, -1),  # height
                            self.freqs[2][:w].view(1, 1, w, -1).expand(f-cond_lens, h, w, -1)   # width
                        ], dim=-1).reshape(content_seq_len, 1, -1).to(x.device)
            freqs = torch.cat([freqs_ref, freqs_content], dim=0).to(x.device)
            return freqs
        freqs = recompute_freqs(f, h, w, cond_lens)

        ref = x[:, :, :cond_lens]
        x = x[:, :, cond_lens:]
        # Animate
        motion_vec = None
        if pose_latents is not None:
            x, motion_vec = self.after_patch_embedding(x, pose_latents, face_pixel_values)
        x = torch.cat([ref, x], dim=2)
        
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()


        # process timestep embedding
        if timestep.dim() == 1:  # only suitable for tsf
            import os
            if os.environ.get("DEBUG", 0) == 1:
                print("timestep dim is 1, using uniform timestep embedding for all tokens in the batch")
                
            # for content tokens
            t = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))  # (B, dim)
            t_mod = self.time_projection(t).unflatten(1, (6, self.dim))         # (B, 6, dim)

            # for ref tokens: explicit t=0 embedding (consistent with dim==2 branch)
            timestep_0 = torch.zeros_like(timestep)
            t_0 = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep_0).to(x.dtype))  # (B, dim)
            cond_t_mod = self.time_projection(t_0).unflatten(1, (6, self.dim))  # (B, 6, dim)
            t_for_head = t  # (B, dim)

        elif timestep.dim() == 2: # suitable for df and tf (use token-level timestep modulation)

            B, T = timestep.shape

            # noisy content tokens: use actual timestep embedding
            t_flat = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).to(x.dtype)
            )  # (B*T, dim)
            _t_mod = self.time_projection(t_flat).unflatten(1, (6, self.dim))  # (B*T, 6, dim)
            t_mod = _t_mod.repeat_interleave(frame_seqlen, dim=0)  # (B*T*H*W, 6, dim)

            # ref tokens: explicit t=0 embedding (one per batch item)
            timestep_0 = torch.zeros(B, dtype=timestep.dtype, device=timestep.device)
            t_flat_0 = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep_0).to(x.dtype))  # (B, dim)
            _cond_t_mod = self.time_projection(t_flat_0).unflatten(1, (6, self.dim))  # (B, 6, dim)
            cond_t_mod = _cond_t_mod.repeat_interleave(frame_seqlen, dim=0)  # (B*frame_seqlen, 6, dim)

            t_content = t_flat.repeat_interleave(frame_seqlen, dim=0)  # (B*T*H*W, dim)
            t_for_head = t_content

            if clean_x is not None and clean_x is not False:
                # TF mode: clean content tokens also use explicit t=0 embedding
                clean_timestep_0 = torch.zeros(B * T, dtype=timestep.dtype, device=timestep.device)
                clean_t_flat_0 = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, clean_timestep_0).to(x.dtype))  # (B*T, dim)
                _clean_t_mod = self.time_projection(clean_t_flat_0).unflatten(1, (6, self.dim))  # (B*T, 6, dim)
                clean_t_mod = _clean_t_mod.repeat_interleave(frame_seqlen, dim=0)  # (B*T*H*W, 6, dim)
                # [clean_content || noisy_content] concatenated t_mod, matches x layout in TF mode
                t_mod = torch.cat([clean_t_mod, t_mod], dim=0)  # (2*B*T*H*W, 6, dim)
        
        
        # ✅ 构造 block mask
        num_frames = f
        frame_seqlen = h * w
        tf_mode = clean_x is not None and clean_x is not False
        if tf_mode:
            mask_key = ("tf", num_frames, h, w, self.num_frame_per_block, cond_lens, self.local_attn_size)
        else:
            mask_key = ("sft", num_frames, h, w, self.num_frame_per_block, cond_lens)

        if mask_key not in self.block_mask_cache:
            if tf_mode:
                self.block_mask_cache[mask_key] = self._prepare_teacher_forcing_mask_i2v(
                    device=x.device,
                    num_frames=num_frames,
                    frame_seqlen=frame_seqlen,
                    num_frame_per_block=self.num_frame_per_block,
                    ref_seqlen=frame_seqlen * cond_lens,
                    local_attn_size=self.local_attn_size,
                )
            elif kwargs.get('bid_mode', None) is not None and kwargs.get('bid_mode', None) is True:
                self.block_mask_cache[mask_key] = self._prepare_blockwise_causal_attn_mask_i2v_bid(
                    device=x.device,
                    num_frames=num_frames,
                    frame_seqlen=frame_seqlen,
                    num_frame_per_block=self.num_frame_per_block,
                    ref_seqlen=frame_seqlen * cond_lens,
                    local_attn_size=self.local_attn_size,
                )
            else:
                self.block_mask_cache[mask_key] = self._prepare_blockwise_causal_attn_mask_i2v(
                    device=x.device,
                    num_frames=num_frames,
                    frame_seqlen=frame_seqlen,
                    num_frame_per_block=self.num_frame_per_block,
                    ref_seqlen=frame_seqlen * cond_lens,
                    local_attn_size=self.local_attn_size,
                )
        self.block_mask = self.block_mask_cache[mask_key]

        if tf_mode:
            clean_x = torch.cat([clean_x, y], dim=1)
            clean_x, _ = self.patchify(clean_x)

            if pose_latents is not None:
                clean_ref = clean_x[:, :, :cond_lens]
                clean_content = clean_x[:, :, cond_lens:]
                clean_content = clean_content + self.pose_patch_embedding(pose_latents)
                clean_x = torch.cat([clean_ref, clean_content], dim=2)

            clean_x = rearrange(clean_x, 'b c f h w -> b (f h w) c').contiguous()

            ref_tokens = frame_seqlen * cond_lens
            clean_content = clean_x[:, ref_tokens:]
            noisy_content = x[:, ref_tokens:]
            ref = x[:, :ref_tokens]
            x = torch.cat([ref, clean_content, noisy_content], dim=1)
        
        def create_custom_forward(module):
                def custom_forward(*inputs, **kwargs):
                    return module(*inputs, **kwargs)
                return custom_forward

        for block_id, block in enumerate(self.blocks):

            if use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, cond_t_mod, freqs,                 
                            frame_seqlen=frame_seqlen*cond_lens,
                            block_mask=self.block_mask,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, cond_t_mod, freqs,
                        frame_seqlen=frame_seqlen*cond_lens,
                        block_mask=self.block_mask,
                        use_reentrant=False,
                    )
            else:
                x = block(
                    x, context, t_mod, cond_t_mod, freqs, 
                    frame_seqlen=frame_seqlen*cond_lens,
                    block_mask=self.block_mask,
                    )
            
            # Animate
            if pose_latents is not None and face_pixel_values is not None:
                ref = x[:, :frame_seqlen*cond_lens]
                content = x[:, frame_seqlen*cond_lens:]
                if tf_mode:
                    clean_tokens = (f - cond_lens) * frame_seqlen
                    clean_content = content[:, :clean_tokens]
                    noisy_content = content[:, clean_tokens:]
                    clean_content = self.after_transformer_block(block_id, clean_content, motion_vec)
                    noisy_content = self.after_transformer_block(block_id, noisy_content, motion_vec)
                    content = torch.cat([clean_content, noisy_content], dim=1)
                else:
                    content = self.after_transformer_block(block_id, content, motion_vec)
                x = torch.cat([ref, content], dim=1)
        # import pdb; pdb.set_trace()
        if tf_mode:
            ref_tokens = frame_seqlen * cond_lens
            content_tokens = x.shape[1] - ref_tokens
            noisy_start = ref_tokens + content_tokens // 2
            x = x[:, noisy_start:]
        else:
            x = x[:,frame_seqlen*cond_lens:]  # remove reference frame tokens
        x = self.head(x, t_for_head)
        x = self.unpatchify(x, (f-cond_lens, h, w))
        return x
    
    def forward(
                self,
                *args,
                **kwargs
            ):
        if kwargs.get('kv_cache', None) is not None:
            print("kv_cache is provided, run in inference mode.")
            assert False # we currently do not support inference with gradient checkpointing, as it requires careful design of kv_cache to avoid storing large intermediate activations. We will add this support in the future.
        else:
            return self._forward_train(*args, **kwargs)

    def forward_org(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x
    
    def after_patch_embedding(self, x: List[torch.Tensor], pose_latents, face_pixel_values):
        pose_latents = self.pose_patch_embedding(pose_latents)
        x = x + pose_latents

        if face_pixel_values is None:
            return x, None

        b,c,T,h,w = face_pixel_values.shape
        face_pixel_values = rearrange(face_pixel_values, "b c t h w -> (b t) c h w")

        encode_bs = 8
        face_pixel_values_tmp = []
        for i in range(math.ceil(face_pixel_values.shape[0]/encode_bs)):
            face_pixel_values_tmp.append(self.motion_encoder.get_motion(face_pixel_values[i*encode_bs:(i+1)*encode_bs]))

        motion_vec = torch.cat(face_pixel_values_tmp)
        
        motion_vec = rearrange(motion_vec, "(b t) c -> b t c", t=T)
        motion_vec = self.face_encoder(motion_vec)
        return x, motion_vec
    
    def after_transformer_block(self, block_idx, x, motion_vec, motion_masks=None):
        if block_idx % 5 == 0:
            adapter_args = [x, motion_vec, motion_masks, False]
            residual_out = self.face_adapter.fuser_blocks[block_idx // 5](*adapter_args)
            x = residual_out + x
        return x


    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v_bid(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, ref_seqlen=1560, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen # num_frames

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)
        ends[:ref_seqlen] = ref_seqlen # [ ref_seqlen | 0 | 0 | ... ] the first ref_seqlen tokens can attend to each other
        
        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=ref_seqlen, 
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        ) # from ref_seqlen to total_length 

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block # [ condition block (ref_seqlen) | block1 (ref_seqlen + step*1) | block2 (ref_seqlen + step*2) | ... ]

        def attention_mask(b, h, q_idx, kv_idx):
            return (kv_idx < ends[-1])
            # if local_attn_size == -1:
            #     return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            # else:
            #     return (kv_idx < ref_seqlen) | (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, ref_seqlen=1560, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen # num_frames

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)
        ends[:ref_seqlen] = ref_seqlen # [ ref_seqlen | 0 | 0 | ... ] the first ref_seqlen tokens can attend to each other
        
        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=ref_seqlen, 
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        ) # from ref_seqlen to total_length 

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block # [ condition block (ref_seqlen) | block1 (ref_seqlen + step*1) | block2 (ref_seqlen + step*2) | ... ]

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return (kv_idx < ref_seqlen) | ( (kv_idx < ends[q_idx]) & (kv_idx >= ends[q_idx] - frame_seqlen * local_attn_size) ) | (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask_i2v(
        device: torch.device | str,
        num_frames: int = 21,
        frame_seqlen: int = 1560,
        num_frame_per_block: int = 1,
        ref_seqlen: int = 1560,
        local_attn_size: int = -1,
    ) -> BlockMask:
        single_len = num_frames * frame_seqlen
        content_len = single_len - ref_seqlen
        total_length = ref_seqlen + content_len * 2

        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        clean_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noisy_clean_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noisy_clean_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noisy_self_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noisy_self_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        clean_content_start = ref_seqlen
        noisy_content_start = ref_seqlen + content_len

        # ref tokens in clean stream
        clean_context_ends[:ref_seqlen] = ref_seqlen

        # clean content tokens: blockwise causal (with access to ref)
        block_size = frame_seqlen * num_frame_per_block
        clean_content_starts = torch.arange(
            start=clean_content_start,
            end=clean_content_start + content_len,
            step=block_size,
            device=device,
            dtype=torch.long,
        )
        clean_content_ends = torch.clamp(clean_content_starts + block_size, max=clean_content_start + content_len)
        for start, end in zip(clean_content_starts.tolist(), clean_content_ends.tolist()):
            clean_context_ends[start:end] = end
            if local_attn_size == -1:
                clean_context_starts[start:end] = ref_seqlen
            else:
                local_clean_window = frame_seqlen * local_attn_size
                clean_context_starts[start:end] = max(ref_seqlen, end - local_clean_window)

        # noisy content tokens:
        # - attend same noisy block only
        # - attend clean stream up to previous clean block end
        noisy_content_starts = torch.arange(
            start=noisy_content_start,
            end=noisy_content_start + content_len,
            step=block_size,
            device=device,
            dtype=torch.long,
        )
        noisy_content_ends = torch.clamp(noisy_content_starts + block_size, max=noisy_content_start + content_len)

        clean_block_ends = [ref_seqlen]
        clean_block_ends.extend(clean_content_ends.tolist())

        for block_idx, (start, end) in enumerate(zip(noisy_content_starts.tolist(), noisy_content_ends.tolist()), start=1):
            noisy_self_starts[start:end] = start
            noisy_self_ends[start:end] = end
            prev_clean_end = clean_block_ends[block_idx - 1]
            noisy_clean_ends[start:end] = prev_clean_end
            if local_attn_size == -1:
                noisy_clean_starts[start:end] = ref_seqlen
            else:
                # Interpret local_attn_size as total frame window including noisy self block.
                # So noisy->clean context uses (local_attn_size - num_frame_per_block) frames.
                noisy_clean_window_frames = max(local_attn_size - num_frame_per_block, 0)
                noisy_clean_window = frame_seqlen * noisy_clean_window_frames
                noisy_clean_starts[start:end] = max(ref_seqlen, prev_clean_end - noisy_clean_window)

        def attention_mask(b, h, q_idx, kv_idx):
            clean_ref_mask = (q_idx < ref_seqlen) & (kv_idx < ref_seqlen)
            clean_content_mask = (
                (q_idx >= clean_content_start) &
                (q_idx < noisy_content_start) &
                (
                    (kv_idx < ref_seqlen) |
                    ((kv_idx >= clean_context_starts[q_idx]) & (kv_idx < clean_context_ends[q_idx]))
                )
            )
            clean_mask = clean_ref_mask | clean_content_mask

            noisy_mask = (q_idx >= noisy_content_start) & (
                ((kv_idx >= noisy_self_starts[q_idx]) & (kv_idx < noisy_self_ends[q_idx])) |
                (kv_idx < ref_seqlen) |
                ((kv_idx >= noisy_clean_starts[q_idx]) & (kv_idx < noisy_clean_ends[q_idx]))
            )

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noisy_mask

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)
        return block_mask