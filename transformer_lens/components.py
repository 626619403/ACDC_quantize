import logging
from typing import Dict, Optional, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fancy_einsum import einsum
from jaxtyping import Float, Int
from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCacheEntry
from torch import float16,float32
#from transformer_engine.pytorch import fp8_autocast, fp8_quantize_dequantize
#import transformer_engine.pytorch as te
#from transformer_engine.common.recipe import Format
from transformer_lens.utils import (
    gelu_fast,
    gelu_new,
    get_causal_mask_for_left_padding,
    solu,
)
from transformer_lens.GEMM import matmul_device_tma_persistent

class FP8Manager:
    def __init__(self, max_val=448.0):
        pass

    def quantize(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        return tensor.to(torch.float8_e4m3fn)

    def dequantize(self, tensor: torch.Tensor, name: str) -> torch.Tensor:

        return tensor.to(float32)


# Embed & Unembed
class Embed(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_E: Float[torch.Tensor, "d_vocab d_model"] = nn.Parameter(
            torch.empty(self.cfg.d_vocab, self.cfg.d_model, dtype=cfg.dtype)
        )

    def forward(
        self, tokens: Int[torch.Tensor, "batch pos"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
        # B acts as a tensor of indices into the second dimension (so >=0 and <b)
        return self.W_E[tokens, :]


class Unembed(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        # Note that there's a separate variable for d_vocab_out and d_vocab (the input vocab size). For language tasks these are always the same, but for algorithmic tasks we may want them to be different.
        self.W_U: Float[torch.Tensor, "d_model d_vocab_out"] = nn.Parameter(
            torch.empty(self.cfg.d_model, self.cfg.d_vocab_out, dtype=cfg.dtype)
        )
        self.b_U: Float[torch.Tensor, "d_vocab_out"] = nn.Parameter(
            torch.zeros(self.cfg.d_vocab_out, dtype=cfg.dtype)
        )

    def forward(
        self, residual: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_vocab_out"]:
        return (
            einsum(
                "batch pos d_model, d_model vocab -> batch pos vocab",
                residual,
                self.W_U,
            )
            + self.b_U
        )


# Positional Embeddings
class PosEmbed(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_pos = nn.Parameter(
            torch.empty(self.cfg.n_ctx, self.cfg.d_model, dtype=cfg.dtype)
        )

    def forward(
        self,
        tokens: Int[torch.Tensor, "batch pos"],
        past_kv_pos_offset: int = 0,
        left_attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """
        Forward pass for positional embeddings.

        Args:
            tokens (Int[torch.Tensor, "batch pos"]): Input tokens.
            past_kv_pos_offset (int, optional): The length of tokens in the past_kv_cache. Defaults to 0.
            left_attention_mask (Int[torch.Tensor, "batch pos"], optional): The attention mask for left padded tokens.
                None when right padding is used. Defaults to None.

        Returns:
            Float[torch.Tensor, "batch pos d_model"]: Absolute position embeddings.
        """
        tokens_length = tokens.size(-1)

        if left_attention_mask is None:
            # Right padding case
            pos_embed = self.W_pos[
                past_kv_pos_offset : tokens_length + past_kv_pos_offset, :
            ]  # [pos, d_model]
            batch_pos_embed = einops.repeat(
                pos_embed, "pos d_model -> batch pos d_model", batch=tokens.size(0)
            )

        else:
            # Left padding case
            # Separated from the right padding case for computational efficiency
            # (this code is a bit slower than the code above)

            # shift the position ids so that the id at the the first attended token position becomes zero.
            # The position ids of the prepending pad tokens are shifted to -1.
            shifted_position_ids = (
                left_attention_mask.T.cumsum(dim=0) - 1
            )  # [tokens_length, batch]

            # Set the position ids of all prepending pad tokens to an arbitrary number (zero here)
            # just to avoid indexing errors.
            position_ids = shifted_position_ids.masked_fill(shifted_position_ids < 0, 0)
            offsetted_position_ids = position_ids[
                past_kv_pos_offset : tokens_length + past_kv_pos_offset, :
            ]  # [pos, batch]
            pos_embed = self.W_pos[offsetted_position_ids]  # [pos, batch, d_model]

            # Set the position embeddings to 0 for pad tokens
            padding_mask = ~left_attention_mask.T.bool()  # [tokens_length, batch]
            offsetted_padding_mask = padding_mask[
                past_kv_pos_offset : tokens_length + past_kv_pos_offset, :
            ].unsqueeze(
                -1
            )  # [pos, batch, 1]
            batch_pos_embed = torch.where(
                offsetted_padding_mask, 0, pos_embed
            ).transpose(0, 1)

        return batch_pos_embed.clone()


class TokenTypeEmbed(nn.Module):
    """
    The token-type embed is a binary ids indicating whether a token belongs to sequence A or B. For example, for two sentences: "[CLS] Sentence A [SEP] Sentence B [SEP]", token_type_ids would be [0, 0, ..., 0, 1, ..., 1, 1]. `0` represents tokens from Sentence A, `1` from Sentence B. If not provided, BERT assumes a single sequence input. Typically, shape is (batch_size, sequence_length).

    See the BERT paper for more information: https://arxiv.org/pdf/1810.04805.pdf
    """

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_token_type = nn.Parameter(
            torch.empty(2, self.cfg.d_model, dtype=cfg.dtype)
        )

    def forward(self, token_type_ids: Int[torch.Tensor, "batch pos"]):
        return self.W_token_type[token_type_ids, :]


class BertEmbed(nn.Module):
    """
    Custom embedding layer for a BERT-like model. This module computes the sum of the token, positional and token-type embeddings and takes the layer norm of the result.
    """

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.token_type_embed = TokenTypeEmbed(cfg)
        self.ln = LayerNorm(cfg)

        self.hook_embed = HookPoint()
        self.hook_pos_embed = HookPoint()
        self.hook_token_type_embed = HookPoint()

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch pos"],
        token_type_ids: Optional[Int[torch.Tensor, "batch pos"]] = None,
    ):
        base_index_id = torch.arange(input_ids.shape[1], device=input_ids.device)
        index_ids = einops.repeat(
            base_index_id, "pos -> batch pos", batch=input_ids.shape[0]
        )
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_embeddings_out = self.hook_embed(self.embed(input_ids))
        position_embeddings_out = self.hook_pos_embed(self.pos_embed(index_ids))
        token_type_embeddings_out = self.hook_token_type_embed(
            self.token_type_embed(token_type_ids)
        )

        embeddings_out = (
            word_embeddings_out + position_embeddings_out + token_type_embeddings_out
        )
        layer_norm_out = self.ln(embeddings_out)
        return layer_norm_out


class BertMLMHead(nn.Module):
    """
    Transforms BERT embeddings into logits. The purpose of this module is to predict masked tokens in a sentence.
    """

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W = nn.Parameter(torch.empty(cfg.d_model, cfg.d_model, dtype=cfg.dtype))
        self.b = nn.Parameter(torch.zeros(cfg.d_model, dtype=cfg.dtype))
        self.act_fn = nn.GELU()
        self.ln = LayerNorm(cfg)

    def forward(self, resid: Float[torch.Tensor, "batch pos d_model"]) -> torch.Tensor:
        resid = (
            einsum(
                "batch pos d_model_in, d_model_out d_model_in -> batch pos d_model_out",
                resid,
                self.W,
            )
            + self.b
        )
        resid = self.act_fn(resid)
        resid = self.ln(resid)
        return resid


# LayerNormPre
# I fold the LayerNorm weights and biases into later weights and biases.
# This is just the 'center and normalise' part of LayerNorm
# Centering is equivalent to just deleting one direction of residual space,
# and is equivalent to centering the weight matrices of everything writing to the residual stream
# Normalising is a funkier non-linear operation, that projects the residual stream onto the unit hypersphere
class LayerNormPre(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        """LayerNormPre - the 'center and normalise' part of LayerNorm. Length is
        normally d_model, but is d_mlp for softmax. Not needed as a parameter. This
        should only be used in inference mode after folding in LayerNorm weights"""
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.eps = self.cfg.eps

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos]
        # Hook Normalized captures LN output - here it's a vector with std 1 and mean 0
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(
        self,
        x: Union[
            Float[torch.Tensor, "batch pos d_model"],
            Float[torch.Tensor, "batch pos head_index d_model"],
        ],
    ) -> Union[
        Float[torch.Tensor, "batch pos d_model"],
        Float[torch.Tensor, "batch pos head_index d_model"],
    ]:
        if self.cfg.dtype not in [torch.float32, torch.float64]:
            x = x.to(torch.float32)

        x = x - x.mean(axis=-1, keepdim=True)  # [batch, pos, length]
        scale: Union[
            Float[torch.Tensor, "batch pos 1"],
            Float[torch.Tensor, "batch pos head_index 1"],
        ] = self.hook_scale((x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt())
        return self.hook_normalized(x / scale).to(self.cfg.dtype)


class LayerNorm(nn.Module):
    def __init__(
        self, cfg: Union[Dict, HookedTransformerConfig], length: Optional[int] = None
    ):
        """
        LayerNorm with optional length parameter

        length (Optional[int]): If the dimension of the LayerNorm. If not provided, assumed to be d_model
        """
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.eps = self.cfg.eps
        if length is None:
            self.length = self.cfg.d_model
        else:
            self.length = length

        self.w = nn.Parameter(torch.ones(self.length, dtype=cfg.dtype))
        self.b = nn.Parameter(torch.zeros(self.length, dtype=cfg.dtype))

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos, 1]
        # Hook_normalized is on the LN output
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(
        self,
        x: Union[
            Float[torch.Tensor, "batch pos d_model"],
            Float[torch.Tensor, "batch pos head_index d_model"],
        ],
    ) -> Union[
        Float[torch.Tensor, "batch pos d_model"],
        Float[torch.Tensor, "batch pos head_index d_model"],
    ]:
        if self.cfg.dtype not in [torch.float32, torch.float64]:
            x = x.to(torch.float32)

        x = x - x.mean(axis=-1, keepdim=True)  # [batch, pos, length]
        scale: Float[torch.Tensor, "batch pos 1"] = self.hook_scale(
            (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        )
        x = x / scale  # [batch, pos, length]
        return self.hook_normalized(x * self.w + self.b).to(self.cfg.dtype)


class RMSNormPre(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        """RMSNormPre - LayerNormPre without the centering and bias (RMS = Root Mean Square)"""
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.eps = self.cfg.eps

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos]
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(
        self, x: Float[torch.Tensor, "batch pos length"]
    ) -> Float[torch.Tensor, "batch pos length"]:
        if self.cfg.dtype not in [torch.float32, torch.float64]:
            x = x.to(torch.float32)

        scale: Float[torch.Tensor, "batch pos 1"] = self.hook_scale(
            (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        )
        return self.hook_normalized(x / scale).to(
            self.cfg.dtype
        )  # [batch, pos, length]


class RMSNorm(nn.Module):
    def __init__(
        self, cfg: Union[Dict, HookedTransformerConfig], length: Optional[int] = None
    ):
        """
        RMSNorm - LayerNorm without the centering and bias (RMS = Root Mean Square)

        length (Optional[int]): If the dimension of the RMSNorm. If not provided, assumed to be d_model
        """
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.eps = self.cfg.eps
        if length is None:
            self.length = self.cfg.d_model
        else:
            self.length = length

        self.w = nn.Parameter(torch.ones(self.length, dtype=cfg.dtype))

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos, 1]
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(
        self, x: Float[torch.Tensor, "batch pos length"]
    ) -> Float[torch.Tensor, "batch pos length"]:
        if self.cfg.dtype not in [torch.float32, torch.float64]:
            x = x.to(torch.float32)

        scale: Float[torch.Tensor, "batch pos 1"] = self.hook_scale(
            (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        )
        x = self.hook_normalized(x / scale).to(self.cfg.dtype)  # [batch, pos, length]
        return x * self.w


# 
        
class Attention(nn.Module):
    
    def __init__(
        self,
        cfg: Union[Dict, HookedTransformerConfig],
        attn_type: str = "global",
        layer_id: Optional[int] = None,):
        """
        Attention module supporting mixed high and low precision computation, with hooks,
        rotary positional embeddings, etc., and supporting FP8 quantization.
        """
        super().__init__()
        self.cfg = cfg
        self.attn_type = attn_type
        self.layer_id = layer_id
        self.fp8_manager_q = FP8Manager()
        self.fp8_manager_k = FP8Manager()
        self.fp8_manager_v = FP8Manager()
        
        # Initialize weights
        self.register_parameter('W_Q', nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head), requires_grad=False))
        self.register_parameter('W_K', nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head), requires_grad=False))
        self.register_parameter('W_V', nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head), requires_grad=False))
        self.register_parameter('W_O', nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head, cfg.d_model), requires_grad=False))
        self.register_parameter('b_Q', nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head), requires_grad=False))
        self.register_parameter('b_K', nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head), requires_grad=False))
        self.register_parameter('b_V', nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head), requires_grad=False))
        self.register_parameter('b_O', nn.Parameter(torch.zeros(cfg.d_model), requires_grad=False))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hooks
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn_scores = HookPoint()
        self.hook_pattern = HookPoint()
        self.hook_result = HookPoint()
        


        # Causal mask
        causal_mask = torch.tril(torch.ones((cfg.n_ctx, cfg.n_ctx)).bool()).to(self.device)
        if self.attn_type == "global":
            # For global attention, this is a lower triangular matrix (keys <= queries).
            self.register_buffer("mask", causal_mask)
        elif self.attn_type == "local":
            # For local attention, this mask is banded: query - window_size < key <= query.
            assert isinstance(self.cfg.window_size, int)
            self.register_buffer(
                "mask", torch.triu(causal_mask, 1 - self.cfg.window_size)
            )
        else:
            raise ValueError(f"Invalid attention type: {self.attn_type}")
            
        self.register_buffer("IGNORE", torch.tensor(-1e3))
        
        # Attention score scaling factor
        self.attn_scale = np.sqrt(cfg.d_head) if cfg.use_attn_scale else 1.0
        if cfg.scale_attn_by_inverse_layer_idx:
            self.attn_scale *= (layer_id + 1)
        self.layer_id = layer_id        
        
        # Rotary positional embeddings
        if self.cfg.positional_embedding_type == "shortformer":
            # Tracks the input to keys and queries, i.e. resid_pre + pos_embeds.
            self.hook_attn_input = HookPoint()  # [batch, pos, d_model]
        elif self.cfg.positional_embedding_type == "rotary":
            # Applies a rotation to each two-element chunk of keys and queries
            # before dot-product to bake in relative position information.
            self.hook_rot_k = HookPoint()
            self.hook_rot_q = HookPoint()
            sin, cos = self.calculate_sin_cos_rotary(
                self.cfg.rotary_dim, self.cfg.n_ctx, dtype=self.cfg.dtype
            )
            self.register_buffer("rotary_sin", sin)
            self.register_buffer("rotary_cos", cos)

        
    def forward(
        self,
        query_input: Union[
            Float[torch.Tensor, "batch pos d_model"],
            Float[torch.Tensor, "batch pos n_heads d_model"],
        ],
        key_input: Union[
            Float[torch.Tensor, "batch pos d_model"],
            Float[torch.Tensor, "batch pos n_heads d_model"],
        ],
        value_input: Union[
            Float[torch.Tensor, "batch pos d_model"],
            Float[torch.Tensor, "batch pos n_heads d_model"],
        ],
        past_kv_cache_entry: Optional[HookedTransformerKeyValueCacheEntry] = None,
        additive_attention_mask: Optional[Float[torch.Tensor, "batch 1 1 pos"]] = None,
        left_attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
        selected_head: int = -1, 
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """
        If positional_embedding_type is "shortformer", shortformer_pos_embed is used; otherwise, it is None.
        past_kv_cache_entry is an optional cache of past keys and values, relevant only for text generation.
        additive_attention_mask is an optional mask to add to the attention weights.
        left_attention_mask is used with left-padded tokens (None when right-padding is applied).
        """

        # Determine einsum string format based on whether QKV inputs are split.
        try:
            x=self.W_Q_low
        except AttributeError:
            self.update_W_low()
        stream_load_weights = torch.cuda.Stream()  # 用于加载高精度权重
        stream_compute_high = torch.cuda.Stream()
        if self.cfg.use_split_qkv_input:
            query_input = query_input[:, :, 0, :].squeeze(2)
            key_input   = key_input[:, :, 0, :].squeeze(2)
            value_input = value_input[:, :, 0, :].squeeze(2)
            
        if selected_head!=-1:
            with torch.cuda.stream(stream_load_weights):
                W_Q_high, W_K_high,W_V_high= self.load_high_precision_weights(selected_head)
            
            # Input shape: [batch, pos, d_model]
            # Directly compute high-precision results with output shape [batch, pos, d_head],
            # then unsqueeze to obtain shape [batch, n_heads, pos, d_head].
 
        query_flat = query_input.to(torch.float8_e4m3fn)  
        # 普通矩阵乘法: [batch*pos, d_model] @ [d_model, n_heads*d_head]
        q_low_flat = matmul_device_tma_persistent(
            query_flat.view(-1, self.cfg.d_model),
            self.W_Q_low.view(self.cfg.d_model, -1),
            4
        ) # [batch*pos, n_heads, d_head]
        if selected_head!=-1:
            stream_load_weights.synchronize()
            with torch.cuda.stream(stream_compute_high):
                q_high,k_high,v_high=self.compute_high_precision_results(
                    query_input,key_input,value_input,
                    W_Q_high,W_K_high,W_V_high)
        # Restore shape to [batch, pos, n_heads, d_head] and then permute to [batch, n_heads, pos, d_head]
        q_low = q_low_flat.view(query_input.size(0), query_input.size(1),
                                self.cfg.n_heads, self.cfg.d_head).permute(0, 2, 1, 3)

        key_flat = key_input.to(torch.float8_e4m3fn)   
        k_low_flat = matmul_device_tma_persistent(
            key_flat.view(-1, self.cfg.d_model),  # [batch*pos, d_model]
            self.W_K_low.view(self.cfg.d_model, -1) ,
            4
        )  # [batch*pos, n_heads, d_head]

        # 恢复形状并 permute
        k_low = k_low_flat.view(key_input.size(0), key_input.size(1),  # [batch, pos, n_heads, d_head]
                                self.cfg.n_heads, self.cfg.d_head).permute(0, 2, 1, 3)  # [batch, n_heads, pos, d_head]


        # Value (V) 的处理
        value_flat = value_input.to(torch.float8_e4m3fn) 
        v_low_flat = matmul_device_tma_persistent(
            value_flat.view(-1, self.cfg.d_model),  # [batch*pos, d_model]
            self.W_V_low.view(self.cfg.d_model, -1),
            4  
        )  # [batch*pos, n_heads, d_head]

        # 恢复形状并 permute
        v_low = v_low_flat.view(value_input.size(0), value_input.size(1),  # [batch, pos, n_heads, d_head]
                                self.cfg.n_heads, self.cfg.d_head).permute(0, 2, 1, 3)  # [batch, n_heads, pos, d_head]

        
        q_low = self.fp8_manager_q.dequantize(q_low, 'W_Q')
        k_low = self.fp8_manager_k.dequantize(k_low, 'W_K')
        v_low = self.fp8_manager_v.dequantize(v_low, 'W_V')
        
        # Replace the corresponding selected head in the low-precision branch with the high-precision result.
        stream_compute_high.synchronize()            
        if selected_head!=-1:  # Here, q_low, k_low, and v_low all have shape [batch, n_heads, pos, d_head].
            q_low[:, selected_head:selected_head+1] = q_high
            k_low[:, selected_head:selected_head+1] = k_high
            v_low[:, selected_head:selected_head+1] = v_high
            
        q = q_low.permute(0, 2, 1, 3)
        k = k_low.permute(0, 2, 1, 3)
        v = v_low.permute(0, 2, 1, 3)
        # Add the respective biases and pass through downstream hooks,
        # maintaining the tensor shape [batch, n_heads, pos, d_head].
        v = self.hook_v(v + self.b_V)
        q = self.hook_q(q + self.b_Q)
        k = self.hook_k(k + self.b_K)

        if past_kv_cache_entry is not None:
            # Appends the new keys and values to the cached values, and automatically updates the cache
            kv_cache_pos_offset = past_kv_cache_entry.past_keys.size(1)
            k, v = past_kv_cache_entry.append(k, v)
        else:
            # Not using a cache
            kv_cache_pos_offset = 0

        if self.cfg.positional_embedding_type == "rotary":
            q, k = self.rotary_rotate_qk(q, k, kv_cache_pos_offset)

        if self.cfg.dtype not in [torch.float32, torch.float64]:
            # If using 16 bits, increase the precision to avoid numerical instabilities
            q = q.to(torch.float32)
            k = k.to(torch.float32)

        attn_scores = (
            einsum(
                "batch query_pos head_index d_head, \
                    batch key_pos head_index d_head \
                    -> batch head_index query_pos key_pos",
                q,
                k,
            )
            / self.attn_scale
        )  # [batch, head_index, query_pos, key_pos]
        if self.cfg.attention_dir == "causal":
            # If causal attention, we mask it to only attend backwards. If bidirectional, we don't mask.
            attn_scores = self.apply_causal_mask(
                attn_scores, kv_cache_pos_offset, left_attention_mask
            )  # [batch, head_index, query_pos, key_pos]
        if additive_attention_mask is not None:
            attn_scores += additive_attention_mask

        attn_scores = self.hook_attn_scores(attn_scores)
        pattern = self.hook_pattern(
            F.softmax(attn_scores, dim=-1)
        )  # [batch, head_index, query_pos, key_pos]
        pattern = pattern.to(self.cfg.dtype)
        z = self.hook_z(
            einsum(
                "batch key_pos head_index d_head, \
                batch head_index query_pos key_pos -> \
                batch query_pos head_index d_head",
                v,
                pattern,
            )
        )  # [batch, pos, head_index, d_head]
        if not self.cfg.use_attn_result:
            out = (
                (
                    einsum(
                        "batch pos head_index d_head, \
                            head_index d_head d_model -> \
                            batch pos d_model",
                        z,
                        self.W_O,
                    )
                )
                + self.b_O
            )  # [batch, pos, d_model]
        else:
            # Explicitly calculate the attention result so it can be accessed by a hook
            # This is off by default because it can easily eat through your GPU memory.
            result = self.hook_result(
                einsum(
                    "batch pos head_index d_head, \
                        head_index d_head d_model -> \
                        batch pos head_index d_model",
                    z,
                    self.W_O,
                )
            )  # [batch, pos, head_index, d_model]
            out = (
                einops.reduce(
                    result, "batch position index model->batch position model", "sum"
                )
                + self.b_O
            )  # [batch, pos, d_model]
        return out.to(self.device)
    
    @property
    def OV(self) -> FactoredMatrix:
        """
        OV-Circuit, as defined in A Mathematical Framework. Because there's no non-linearity between the value vector and the output of the layer, the output is purely determined by the matrix W_OV = W_V @ W_O, and not W_V or W_O individually. (Mathematically, for a single head, output == pattern @ residual @ W_V @ W_O, see the glossary for more)

        Done in the order W_V, W_O because the paper uses left-multiplying weight matrices, and TransformerLens uses right-multiplying, sorry!

        Returns a FactoredMatrix, with left matrix W_V [head_index, d_model, d_head] and right matrix W_O [head_index, d_head, d_model] - this is a low rank factorisation of the underlying [head_index, d_model, d_model]. FactoredMatrix has helper functions to deal with these large matrices efficiently. To get the OV circuit of a head k, attn.OV[k] works.
        """
        
        return FactoredMatrix(self.W_V, self.W_O)
    
    @property
    def QK(self) -> FactoredMatrix:
        """
        QK-Circuit, as defined in A Mathematical Framework. Because there's no non-linearity in the key-query dot product, the output is purely determined by the matrix W_QK = W_Q.T @ W_K, and not W_Q or W_K individually. (Mathematically, for a single head, pattern = destination_residual.T @ W_Q.T @ W_K @ source-residual, see the glossary for more).

        Done in the order Q on the left, K on the right, because the pattern has dimensions [destination_pos, source_pos]

        Returns a FactoredMatrix, with left matrix W_Q [head_index, d_model, d_head] and right matrix W_K.T [head_index, d_head, d_model] - this is a low rank factorisation of the underlying [head_index, d_model, d_model] matrix. FactoredMatrix has helper functions to deal with these large matrices efficiently. To get the QK circuit of a head k, attn.QK[k] works.
        """
        W_K_transpose = einops.rearrange(
            self.W_K, "head_index d_model d_head -> head_index d_head d_model"
        )
        return FactoredMatrix(self.W_Q, W_K_transpose)
    
    def update_W_low(self):
        with torch.no_grad():
            W_Q_low = self.fp8_manager_q.quantize(self.W_Q.to('cuda'), 'W_Q').permute(1, 0, 2).contiguous()
            self.register_buffer('W_Q_low', W_Q_low)
            W_K_low = self.fp8_manager_k.quantize(self.W_K.to('cuda'), 'W_K').permute(1, 0, 2).contiguous()
            self.register_buffer('W_K_low', W_K_low)
            W_V_low = self.fp8_manager_v.quantize(self.W_V.to('cuda'), 'W_V').permute(1, 0, 2).contiguous()
            self.register_buffer('W_V_low', W_V_low)
            b_Q_low = self.fp8_manager_q.quantize(self.b_Q.to(self.device), 'b_Q')
            self.register_buffer('b_Q_low', b_Q_low)
            b_K_low = self.fp8_manager_q.quantize(self.b_K.to(self.device), 'b_K')
            self.register_buffer('b_K_low', b_K_low)
            b_V_low = self.fp8_manager_q.quantize(self.b_V.to(self.device), 'b_V')
            self.register_buffer('b_V_low', b_V_low)
            
    def apply_causal_mask(
        self,
        attn_scores: Float[
            torch.Tensor, "batch head_index pos pos_plus_past_kv_pos_offset"
        ],
        past_kv_pos_offset: int = 0,
        left_attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
    ):
        # The query context length is the number of positions we take queries from - if not using a past_kv_cache this is just the context length (for the current prompt), but if we're caching it's just a single token.
        query_ctx_length = attn_scores.size(-2)
        # The key context length is the number of positions in the past - this includes all positions in the cache
        # If not caching, query_ctx_length == key_ctx_length
        key_ctx_length = attn_scores.size(-1)

        assert (
            query_ctx_length + past_kv_pos_offset == key_ctx_length
        ), f"query_ctx_length {query_ctx_length} + past_kv_pos_offset {past_kv_pos_offset} != key_ctx_length {key_ctx_length} - you likely have a bug."

        if left_attention_mask is None:
            # Right padding case
            # Apply only a causal mask to the attention scores
            final_mask = self.mask[None, None]  # [1, 1, pos, pos]
        else:
            # Left padding case
            # Apply a causal mask to the attention scores considering the left padding
            final_mask = get_causal_mask_for_left_padding(left_attention_mask)
            final_mask = final_mask.unsqueeze(1).to(
                attn_scores.device
            )  # [batch, 1, pos, pos]

        masked_attn_scores = torch.where(
            final_mask[
                :,
                :,
                past_kv_pos_offset : past_kv_pos_offset + query_ctx_length,
                :key_ctx_length,
            ],
            attn_scores,
            self.IGNORE,
        )

        # Return the masked attention scores
        return masked_attn_scores

    def rotary_rotate_qk(
        self,
        q: Float[torch.Tensor, "batch q_pos head_index d_head"],
        k: Float[torch.Tensor, "batch k_pos head_index d_head"],
        past_kv_pos_offset,
    ) -> Tuple[
        Float[torch.Tensor, "batch q_pos head_index d_head"],
        Float[torch.Tensor, "batch k_pos head_index d_head"],
    ]:
        # We first apply standard q and k calculation
        q = self.hook_rot_q(self.apply_rotary(q, past_kv_pos_offset))
        k = self.hook_rot_k(self.apply_rotary(k))
        return q, k

    def calculate_sin_cos_rotary(
        self,
        rotary_dim: int,
        n_ctx: int,
        base: int = 10000,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[
        Float[torch.Tensor, "n_ctx rotary_dim"], Float[torch.Tensor, "n_ctx rotary_dim"]
    ]:
        """
        Calculate the sine and cosine waves to use in a rotary embedding. See https://blog.eleuther.ai/rotary-embeddings/ for details

        Note: For some inexplicable reason, in GPT-J each ADJACENT pair of elements in k and q are rotated, in GPT-NeoX the pair of elements at k and k+n//2 are rotated (ie folding the full length in half, and then looking at pairs accordingly). I have absolutely no clue why, it should be completely equivalent.
        To resolve this, I've coded it to default to the GPT-J mode, but to explicitly check whether it's GPT-NeoX and then do the GPT-NeoX thing if it is.
        """
        high_precision = torch.float32 if dtype != torch.float64 else torch.float64
        pos = torch.arange(n_ctx, dtype=high_precision)
        dim = torch.arange(rotary_dim // 2, dtype=high_precision)

        # A set of frequencies evenly spaced in log space
        freq = base ** (dim / (rotary_dim / 2))
        if self.cfg.original_architecture in ["GPTNeoXForCausalLM", "LlamaForCausalLM"]:
            freq = einops.repeat(freq, "d -> (2 d)")
        else:
            freq = einops.repeat(freq, "d -> (d 2)")
        # Create a n_ctx x rotary_dim tensor, where each column is an arithmetic sequence of angles in that frequency
        angles = pos[:, None] / freq[None, :]
        return torch.sin(angles).to(dtype), torch.cos(angles).to(dtype)

    def rotate_every_two(
        self, x: Float[torch.Tensor, "... rotary_dim"]
    ) -> Float[torch.Tensor, "... rotary_dim"]:
        """
        Rotary helper function, splits x into blocks of size 2 along the final axis and maps [x0, x1] to [-x1, x0]

        The final axis of x must have even length.

        GPT-NeoX and GPT-J do rotary subtly differently, see calculate_sin_cos_rotary for details.
        """
        rot_x = x.clone()
        if self.cfg.original_architecture in ["GPTNeoXForCausalLM", "LlamaForCausalLM"]:
            n = x.size(-1) // 2
            rot_x[..., :n] = -x[..., n:]
            rot_x[..., n:] = x[..., :n]
        else:
            rot_x[..., ::2] = -x[..., 1::2]
            rot_x[..., 1::2] = x[..., ::2]

        return rot_x

    def apply_rotary(
        self,
        x: Float[torch.Tensor, "batch pos head_index d_head"],
        past_kv_pos_offset=0,
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        # Only apply rotary to first rotary_dim dimensions (eg, if rotary_dim=64 and d_head=256, only apply to first 1/4 of dimensions)
        x_pos = x.size(1)
        x_rot = x[..., : self.cfg.rotary_dim]
        x_pass = x[..., self.cfg.rotary_dim :]
        x_flip = self.rotate_every_two(x_rot)
        x_rotated = (
            x_rot
            * self.rotary_cos[past_kv_pos_offset : past_kv_pos_offset + x_pos, None, :]
            + x_flip
            * self.rotary_sin[past_kv_pos_offset : past_kv_pos_offset + x_pos, None, :]
        )
        return torch.cat([x_rotated, x_pass], dim=-1)
    
    def load_high_precision_weights(self, selected_head):
        
        W_Q_high = self.W_Q[selected_head].to(self.device)  # [d_model, d_head]
        W_K_high = self.W_K[selected_head].to(self.device)  # [d_model, d_head]
        W_V_high = self.W_V[selected_head].to(self.device)  # [d_model, d_head]

        return W_Q_high, W_K_high, W_V_high
    
    def compute_high_precision_results(self,
                query_input, key_input, value_input,W_Q_high,W_K_high,W_V_high):
        q_high = einsum("batch pos d_model, d_model d_head-> batch pos d_head",
                                query_input.to(self.device), W_Q_high).unsqueeze(1)
        k_high = einsum("batch pos d_model, d_model d_head-> batch pos d_head",
                                key_input.to(self.device),   W_K_high).unsqueeze(1)
        v_high = einsum("batch pos d_model, d_model d_head-> batch pos d_head",
                                value_input.to(self.device), W_V_high).unsqueeze(1)
        return q_high,k_high,v_high
           

# MLP Layers
class MLP(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_in = nn.Parameter(
            torch.empty(self.cfg.d_model, self.cfg.d_mlp, dtype=cfg.dtype)
        )
        self.b_in = nn.Parameter(torch.zeros(self.cfg.d_mlp, dtype=cfg.dtype))
        self.W_out = nn.Parameter(
            torch.empty(self.cfg.d_mlp, self.cfg.d_model, dtype=cfg.dtype)
        )
        self.b_out = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=cfg.dtype))

        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        self.hook_post = HookPoint()  # [batch, pos, d_mlp]

        if self.cfg.act_fn == "relu":
            self.act_fn = F.relu
        elif self.cfg.act_fn == "gelu":
            self.act_fn = F.gelu
        elif self.cfg.act_fn == "silu":
            self.act_fn = F.silu
        elif self.cfg.act_fn == "gelu_new":
            self.act_fn = gelu_new
        elif self.cfg.act_fn == "gelu_fast":
            self.act_fn = gelu_fast
        elif self.cfg.act_fn == "solu_ln":
            self.act_fn = solu
            # Hook taken between activation and layer norm
            self.hook_mid = HookPoint()  # [batch, pos, d_mlp]
            if self.cfg.normalization_type == "LN":
                self.ln = LayerNorm(self.cfg, self.cfg.d_mlp)
            else:
                self.ln = LayerNormPre(self.cfg)

        else:
            raise ValueError(f"Invalid activation function name: {self.cfg.act_fn}")

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # Technically, all these einsums could be done with a single matmul, but this is more readable.
        pre_act = self.hook_pre(
            einsum("batch pos d_model, d_model d_mlp -> batch pos d_mlp", x, self.W_in)
            + self.b_in
        )  # [batch, pos, d_mlp]
        if not self.cfg.act_fn.endswith("_ln"):
            post_act = self.hook_post(self.act_fn(pre_act))  # [batch, pos, d_mlp]
        else:
            mid_act = self.hook_mid(self.act_fn(pre_act))  # [batch, pos, d_mlp]
            post_act = self.hook_post(self.ln(mid_act))
        return (
            einsum(
                "batch pos d_mlp, d_mlp d_model -> batch pos d_model",
                post_act,
                self.W_out,
            )
            + self.b_out
        )



# TODO
# not sure whether to fold this into MLP or not
class GatedMLP(nn.Module):
    """
    The equation of a gated MLP:
    pre = x @ W_gate
    pre_linear = x @ W_in
    post = Gelu(pre) * (pre_linear) + b_in
    mlp_out = post @ W_out + b_out

    In one equation, mlp_out = (Gelu(x @ W_gate) * (x @ W_in) + b_in) @ W_out + b_out
    """

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_in = nn.Parameter(
            torch.empty(self.cfg.d_model, self.cfg.d_mlp, dtype=cfg.dtype)
        )
        self.W_gate = nn.Parameter(
            torch.empty(self.cfg.d_model, self.cfg.d_mlp, dtype=cfg.dtype)
        )
        self.b_in = nn.Parameter(torch.zeros(self.cfg.d_mlp, dtype=cfg.dtype))
        self.W_out = nn.Parameter(
            torch.empty(self.cfg.d_mlp, self.cfg.d_model, dtype=cfg.dtype)
        )
        self.b_out = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=cfg.dtype))

        # hook on gate output but before act_fn
        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        # hook on the linear component of the input
        self.hook_pre_linear = HookPoint()  # [batch, pos, d_mlp]
        # hook on act_fn(gate_output) * W_in(x) + b_in
        self.hook_post = HookPoint()  # [batch, pos, d_mlp]

        if self.cfg.act_fn == "relu":
            self.act_fn = F.relu
        elif self.cfg.act_fn == "gelu":
            self.act_fn = F.gelu
        elif self.cfg.act_fn == "silu":
            self.act_fn = F.silu
        elif self.cfg.act_fn == "gelu_new":
            self.act_fn = gelu_new
        elif self.cfg.act_fn == "gelu_fast":
            self.act_fn = gelu_fast
        elif self.cfg.act_fn == "solu_ln":
            self.act_fn = solu
            # Hook taken between activation and layer norm
            self.hook_mid = HookPoint()  # [batch, pos, d_mlp]
            if self.cfg.normalization_type == "LN":
                self.ln = LayerNorm(self.cfg, self.cfg.d_mlp)
            else:
                self.ln = LayerNormPre(self.cfg)

        else:
            raise ValueError(f"Invalid activation function name: {self.cfg.act_fn}")

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # Technically, all these einsums could be done with a single matmul, but this is more readable.
        pre_act = self.hook_pre(
            einsum(
                "batch pos d_model, d_model d_mlp -> batch pos d_mlp", x, self.W_gate
            )
        )  # [batch, pos, d_mlp]
        if not self.cfg.act_fn.endswith("_ln"):
            pre_linear = self.hook_pre_linear(
                einsum(
                    "batch pos d_model, d_model d_mlp -> batch pos d_mlp", x, self.W_in
                )
            )
            post_act = self.hook_post(
                (self.act_fn(pre_act) * pre_linear) + self.b_in
            )  # [batch, pos, d_mlp]
        else:
            mid_act = self.hook_mid(self.act_fn(pre_act))  # [batch, pos, d_mlp]
            post_act = self.hook_post(self.ln(mid_act))
        return (
            einsum(
                "batch pos d_mlp, d_mlp d_model -> batch pos d_model",
                post_act,
                self.W_out,
            )
            + self.b_out
        )


# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig], block_index):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.selected_head = -1  # Default to no specific head selected

        # Initialize LayerNorm
        if self.cfg.normalization_type == "LN":
            self.ln1 = LayerNorm(cfg)
            if not self.cfg.attn_only:
                self.ln2 = LayerNorm(cfg)
        elif self.cfg.normalization_type == "LNPre":
            self.ln1 = LayerNormPre(cfg)
            if not self.cfg.attn_only:
                self.ln2 = LayerNormPre(cfg)
        elif self.cfg.normalization_type == "RMS":
            self.ln1 = RMSNorm(cfg)
            if not self.cfg.attn_only:
                self.ln2 = RMSNorm(cfg)
        elif self.cfg.normalization_type == "RMSPre":
            self.ln1 = RMSNormPre(cfg)
            if not self.cfg.attn_only:
                self.ln2 = RMSNormPre(cfg)
        elif self.cfg.normalization_type is None:
            self.ln1 = nn.Identity()
            if not self.cfg.attn_only:
                self.ln2 = nn.Identity()
        else:
            # Use logging.warning for flexibility instead of raising an error
            logging.warning(
                f"Invalid normalization_type passed in {self.cfg.normalization_type}"
            )

        # Initialize the Attention module
        if not self.cfg.use_local_attn:
            self.attn = Attention(cfg, "global", block_index)
        else:
            assert self.cfg.attn_types is not None
            attn_type = self.cfg.attn_types[block_index]
            self.attn = Attention(cfg, attn_type, block_index)

        # Initialize the MLP module
        if not self.cfg.attn_only:
            if self.cfg.gated_mlp:
                self.mlp = GatedMLP(cfg)
            else:
                self.mlp = MLP(cfg)

        # Define hooks for intermediate states
        self.hook_q_input = HookPoint()  # [batch, pos, d_model]
        self.hook_k_input = HookPoint()  # [batch, pos, d_model]
        self.hook_v_input = HookPoint()  # [batch, pos, d_model]
        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_in = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_out = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        if not self.cfg.attn_only and not self.cfg.parallel_attn_mlp:
            self.hook_resid_mid = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]

    def forward(
        self,
        resid_pre: Float[torch.Tensor, "batch pos d_model"],
        shortformer_pos_embed: Optional[Float[torch.Tensor, "batch pos d_model"]] = None,
        past_kv_cache_entry: Optional[HookedTransformerKeyValueCacheEntry] = None,
        left_attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """
        Forward pass for the Transformer Block, supporting mixed precision computation.

        Args:
            resid_pre (torch.Tensor): Residual stream input, shape [batch, pos, d_model].
            shortformer_pos_embed (torch.Tensor, optional): Shortformer positional embeddings, 
                used only when positional_embedding_type == "shortformer".
            past_kv_cache_entry (HookedTransformerKeyValueCacheEntry, optional): Cached K/V values, 
                used for autoregressive generation.
            left_attention_mask (torch.Tensor, optional): Left padding attention mask. 
                None if using right padding.

        Returns:
            torch.Tensor: Residual stream output, shape [batch, pos, d_model].
        """
        # Hook for residual stream input
        resid_pre = self.hook_resid_pre(resid_pre)

        # Initialize Query, Key, Value
        query_input = resid_pre
        key_input = resid_pre
        value_input = resid_pre

        # If using separate Q/K/V inputs, add head dimension
        if self.cfg.use_split_qkv_input:
            def add_head_dimension(tensor):
                return einops.repeat(
                    tensor, "batch pos d_model -> batch pos n_heads d_model", n_heads=self.cfg.n_heads
                ).clone()

            query_input = self.hook_q_input(add_head_dimension(query_input))
            key_input = self.hook_k_input(add_head_dimension(key_input))
            value_input = self.hook_v_input(add_head_dimension(value_input))

            if shortformer_pos_embed is not None:
                shortformer_pos_embed = add_head_dimension(shortformer_pos_embed)

        # Attention module
        attn_out = self.hook_attn_out(
            self.attn(
                query_input=self.ln1(query_input) + (0.0 if shortformer_pos_embed is None else shortformer_pos_embed),
                key_input=self.ln1(key_input) + (0.0 if shortformer_pos_embed is None else shortformer_pos_embed),
                value_input=self.ln1(value_input),
                past_kv_cache_entry=past_kv_cache_entry,  # Pass cached keys/values
                left_attention_mask=left_attention_mask,  # Pass left attention mask
                selected_head=self.selected_head,  # Specify high-precision attention head
            )
        )

        # Add MLP if not attention-only and no parallel MLP
        if not self.cfg.attn_only and not self.cfg.parallel_attn_mlp:
            resid_mid = self.hook_resid_mid(resid_pre + attn_out)
            mlp_in = resid_mid if not self.cfg.use_hook_mlp_in else self.hook_mlp_in(resid_mid.clone())
            normalized_resid_mid = self.ln2(mlp_in)
            mlp_out = self.hook_mlp_out(self.mlp(normalized_resid_mid))
            resid_post = self.hook_resid_post(resid_mid + mlp_out)
        elif self.cfg.parallel_attn_mlp:
            # Parallel attention and MLP (as in GPT-J)
            normalized_resid_pre_2 = self.ln2(
                resid_pre if not self.cfg.use_hook_mlp_in else self.hook_mlp_in(resid_pre.clone())
            )
            mlp_out = self.hook_mlp_out(self.mlp(normalized_resid_pre_2))
            resid_post = self.hook_resid_post(resid_pre + attn_out + mlp_out)
        else:
            resid_post = self.hook_resid_post(resid_pre + attn_out)

        return resid_post

    def update_selected_head(self, selected_head: int):
        """
        Update the selected attention head for high-precision computation.

        Args:
            selected_head (int): Index of the attention head to select.
        """
        self.selected_head = selected_head


class BertBlock(nn.Module):
    """
    BERT Block. Similar to the TransformerBlock, except that the LayerNorms are applied after the attention and MLP, rather than before.
    """

    def __init__(self, cfg: HookedTransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.attn = Attention(cfg)
        self.ln1 = LayerNorm(cfg)
        self.mlp = MLP(cfg)
        self.ln2 = LayerNorm(cfg)

        self.hook_q_input = HookPoint()  # [batch, pos, d_model]
        self.hook_k_input = HookPoint()  # [batch, pos, d_model]
        self.hook_v_input = HookPoint()  # [batch, pos, d_model]

        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_in = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_out = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_mid = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]
        self.hook_normalized_resid_post = HookPoint()  # [batch, pos, d_model]

    def forward(
        self,
        resid_pre: Float[torch.Tensor, "batch pos d_model"],
        additive_attention_mask: Optional[Float[torch.Tensor, "batch 1 1 pos"]] = None,
    ):
        resid_pre = self.hook_resid_pre(resid_pre)

        query_input = resid_pre
        key_input = resid_pre
        value_input = resid_pre

        if self.cfg.use_split_qkv_input:

            def add_head_dimension(tensor):
                return einops.repeat(
                    tensor,
                    "batch pos d_model -> batch pos n_heads d_model",
                    n_heads=self.cfg.n_heads,
                ).clone()

            query_input = self.hook_q_input(add_head_dimension(query_input))
            key_input = self.hook_k_input(add_head_dimension(key_input))
            value_input = self.hook_v_input(add_head_dimension(value_input))

        attn_out = self.hook_attn_out(
            self.attn(
                query_input,
                key_input,
                value_input,
                additive_attention_mask=additive_attention_mask,
            )
        )
        resid_mid = self.hook_resid_mid(resid_pre + attn_out)

        mlp_in = (
            resid_mid
            if not self.cfg.use_hook_mlp_in
            else self.hook_mlp_in(resid_mid.clone())
        )
        normalized_resid_mid = self.ln1(mlp_in)
        mlp_out = self.hook_mlp_out(self.mlp(normalized_resid_mid))
        resid_post = self.hook_resid_post(normalized_resid_mid + mlp_out)
        normalized_resid_post = self.hook_normalized_resid_post(self.ln2(resid_post))

        return normalized_resid_post
