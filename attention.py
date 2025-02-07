'''
This file is used to replace class:
************YOUR ENVIRONMENT***********/lib/python3.10/site-packages/transformer_lens/components.py/Attention.

HAVEN'T BE COMPLETED.HAVEN'T BE COMPLETED.HAVEN'T BE COMPLETED.HAVEN'T BE COMPLETED.
HAVEN'T BE COMPLETED.HAVEN'T BE COMPLETED.HAVEN'T BE COMPLETED.HAVEN'T BE COMPLETED.

TODO:
1.W_qkv_low is empty. need to generate/update them when W_qkv have been loaded rather than when them are created.

2.need to pass attention head number. 
This need to change file/class/parameter/function:
transformer_lens/HookedTransformer.py/HookedTransformer/forward;
transformer_lens/components.py/TransformerBlock/forward;
AND IN ACDC CODE...

3. need to check device 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import einops
from typing import Optional, Union, Tuple, Dict
from torchtyping import Float, Int
from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.hook_points import HookPoint


from transformer_engine.pytorch import fp8_autocast, fp8_quantize_dequantize
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format



class FP8Manager:
    """Manages FP8 quantization using the Transformer Engine's built-in implementation (simpler than manually storing scales)."""
    def __init__(self, fp8_format=Format.E4M3):
        self.fp8_format = fp8_format

    def quantize(self, tensor: torch.Tensor, name: str = None) -> torch.Tensor:
        """
        Uses TE's fp8_quantize_dequantize without dequantization,
        returns the quantized tensor, with internal scale factors managed by TE.
        """
        # Disable autocast to ensure a fixed FP8 format is used during quantization.
        with fp8_autocast(enabled=False):
            q_tensor = fp8_quantize_dequantize(tensor, fp8_format=self.fp8_format, dequantize=False)
        return q_tensor

    def dequantize(self, tensor: torch.Tensor, name: str = None) -> torch.Tensor:
        """
        Uses TE's fp8_quantize_dequantize with dequantization enabled.
        The quantization factors are managed internally by TE, so no manual scaling is needed.
        """
        with fp8_autocast(enabled=False):
            dq_tensor = fp8_quantize_dequantize(tensor, fp8_format=self.fp8_format, dequantize=True)
        return dq_tensor

    def scale(self, name: str = None) -> float:
        """
        Since the Transformer Engine automatically manages the FP8 scale factors,
        this simply returns 1.0 as a placeholder to ensure a valid scale is passed to fp8_gemm.
        """
        return 1.0

        
class QuantizedAttention(nn.Module):
    
    def __init__(self, cfg, attn_type="global", layer_id=None):
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
        self.fp8_manager_o = FP8Manager()
        # Initialize weights
        self.register_parameter('W_Q', nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head), requires_grad=False))
        self.register_parameter('W_K', nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head), requires_grad=False))
        self.register_parameter('W_V', nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head), requires_grad=False))
        self.register_parameter('W_O', nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head, cfg.d_model), requires_grad=False))
        self.register_parameter('b_Q', nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head), requires_grad=False))
        self.register_parameter('b_K', nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head), requires_grad=False))
        self.register_parameter('b_V', nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head), requires_grad=False))
        self.register_parameter('b_O', nn.Parameter(torch.zeros(cfg.d_model), requires_grad=False))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hooks
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn_scores = HookPoint()
        self.hook_pattern = HookPoint()
        self.hook_result = HookPoint()
        
        with torch.no_grad():
            W_Q_low = self.fp8_manager_q.quantize(self.W_Q.to(device), 'W_Q')
            self.register_buffer('W_Q_low', W_Q_low)
            W_K_low = self.fp8_manager_k.quantize(self.W_K.to(device), 'W_K')
            self.register_buffer('W_K_low', W_K_low)
            W_V_low = self.fp8_manager_v.quantize(self.W_V.to(device), 'W_V')
            self.register_buffer('W_V_low', W_V_low)
            b_Q_low = self.fp8_manager_q.quantize(self.b_Q.to(device), 'b_Q')
            self.register_buffer('b_Q_low', b_Q_low)
            b_K_low = self.fp8_manager_q.quantize(self.b_K.to(device), 'b_K')
            self.register_buffer('b_K_low', b_K_low)
            b_V_low = self.fp8_manager_q.quantize(self.b_V.to(device), 'b_V')
            self.register_buffer('b_V_low', b_V_low)

        # Causal mask
        causal_mask = torch.tril(torch.ones((cfg.n_ctx, cfg.n_ctx)).bool(), device=device)
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
        self.attn_scale = torch.sqrt(cfg.d_head) if cfg.use_attn_scale else 1.0
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
            
    @property
    def OV(self) -> FactoredMatrix:
        """
        OV-Circuit, as defined in "A Mathematical Framework". Since there is no non-linearity
        between the value vector and the layer output, the output is determined by the matrix W_OV = W_V @ W_O,
        not W_V or W_O individually. (Mathematically, for a single head, output == pattern @ residual @ W_V @ W_O)

        Note: This multiplies in the order W_V, W_O because the paper uses left-multiplying weight matrices,
        while TransformerLens uses right-multiplying.

        Returns a FactoredMatrix with left matrix W_V [n_heads, d_model, d_head] and right matrix W_O [n_heads, d_head, d_model],
        representing a low-rank factorization of the underlying [n_heads, d_model, d_model] matrix.
        Access the OV circuit for head k via attn.OV[k].
        """
        return FactoredMatrix(self.W_V, self.W_O)
    
    @property
    def QK(self) -> FactoredMatrix:
        """
        QK-Circuit, as defined in "A Mathematical Framework". Since there is no non-linearity in the key-query dot product,
        the output is determined by the matrix W_QK = W_Q.T @ W_K, not W_Q or W_K individually.
        (Mathematically, for a single head, pattern = destination_residual.T @ W_Q.T @ W_K @ source_residual)

        Note: The multiplication is arranged with Q on the left and K on the right, since the attention pattern has shape [destination_pos, source_pos].

        Returns a FactoredMatrix with left matrix W_Q [n_heads, d_model, d_head] and right matrix W_K.T [n_heads, d_head, d_model],
        representing a low-rank factorization of the underlying [n_heads, d_model, d_model] matrix.
        Access the QK circuit for head k via attn.QK[k].
        """
        W_K_transpose = einops.rearrange(
            self.W_K, "n_heads d_model d_head -> n_heads d_head d_model"
        )
        return FactoredMatrix(self.W_Q, W_K_transpose)
        
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
        selected_head: int = 0, 
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """
        If positional_embedding_type is "shortformer", shortformer_pos_embed is used; otherwise, it is None.
        past_kv_cache_entry is an optional cache of past keys and values, relevant only for text generation.
        additive_attention_mask is an optional mask to add to the attention weights.
        left_attention_mask is used with left-padded tokens (None when right-padding is applied).
        """

        # Determine einsum string format based on whether QKV inputs are split.
        if self.cfg.use_split_qkv_input:
            qkv_einops_string = "batch pos n_heads d_model"
        else:
            qkv_einops_string = "batch pos d_model"

        # Step 1: Load the high-precision weights for the selected head onto the GPU.
        W_Q_high = self.W_Q[selected_head]  # [d_model, d_head]
        W_K_high = self.W_K[selected_head]  # [d_model, d_head]
        W_V_high = self.W_V[selected_head]  # [d_model, d_head]

        if self.cfg.use_split_qkv_input:
            # Input shape: [batch, pos, n_heads, d_model]
            # Extract the slice for the selected head and permute to [batch, n_heads, pos, d_model].
            query_selected = query_input[:, :, selected_head:selected_head+1, :].permute(0, 2, 1, 3)
            key_selected   = key_input[:, :, selected_head:selected_head+1, :].permute(0, 2, 1, 3)
            value_selected = value_input[:, :, selected_head:selected_head+1, :].permute(0, 2, 1, 3)
            
            # Compute the high-precision results using torch.einsum with full dimension names.
            # Einsum equation: "batch,n_heads,pos,d_model, d_model,d_head -> batch,n_heads,pos,d_head"
            q_high = torch.einsum("batch,n_heads,pos,d_model, d_model,d_head-> batch,n_heads,pos,d_head",
                                  query_selected, W_Q_high)
            k_high = torch.einsum("batch,n_heads,pos,d_model, d_model,d_head-> batch,n_heads,pos,d_head",
                                  key_selected,   W_K_high)
            v_high = torch.einsum("batch,n_heads,pos,d_model, d_model,d_head-> batch,n_heads,pos,d_head",
                                  value_selected, W_V_high)
        else:
            # Input shape: [batch, pos, d_model]
            # Directly compute high-precision results with output shape [batch, pos, d_head],
            # then unsqueeze to obtain shape [batch, n_heads, pos, d_head].
            q_high = torch.einsum("batch,pos,d_model, d_model,d_head-> batch,pos,d_head",
                                  query_input, W_Q_high).unsqueeze(1)
            k_high = torch.einsum("batch,pos,d_model, d_model,d_head-> batch,pos,d_head",
                                  key_input,   W_K_high).unsqueeze(1)
            v_high = torch.einsum("batch,pos,d_model, d_model,d_head-> batch,pos,d_head",
                                  value_input, W_V_high).unsqueeze(1)
            # In this case, we assume the high-precision branch has only 1 head.

        # Low-precision branch: Uniform output shape [batch, n_heads, pos, d_head]
        with te.fp8_autocast(enabled=True):
            # Reshape input to [batch*pos, d_model]
            query_flat = query_input.view(-1, self.cfg.d_model)
            # FP8 GEMM: [batch*pos, d_model] @ [d_model, n_heads*d_head]
            q_low_flat = te.fp8_gemm(
                query_flat,
                self.W_Q_low.view(self.cfg.d_model, -1),  # [d_model, n_heads*d_head]
                self.fp8_manager_q.scale('W_Q')
            )
            # Restore shape to [batch, pos, n_heads, d_head] and then permute to [batch, n_heads, pos, d_head]
            q_low = q_low_flat.view(query_input.size(0), query_input.size(1),
                                    self.cfg.n_heads, self.cfg.d_head).permute(0, 2, 1, 3)
            
            key_flat = key_input.view(-1, self.cfg.d_model)
            k_low_flat = te.fp8_gemm(
                key_flat,
                self.W_K_low.view(self.cfg.d_model, -1),
                self.fp8_manager_k.scale('W_K')
            )
            k_low = k_low_flat.view(key_input.size(0), key_input.size(1),
                                    self.cfg.n_heads, self.cfg.d_head).permute(0, 2, 1, 3)
            
            value_flat = value_input.view(-1, self.cfg.d_model)
            v_low_flat = te.fp8_gemm(
                value_flat,
                self.W_V_low.view(self.cfg.d_model, -1),
                self.fp8_manager_v.scale('W_V')
            )
            v_low = v_low_flat.view(value_input.size(0), value_input.size(1),
                                    self.cfg.n_heads, self.cfg.d_head).permute(0, 2, 1, 3)
            
        # Dequantize to FP32.
        q_low = self.fp8_manager_q.dequantize(q_low, 'W_Q')
        k_low = self.fp8_manager_k.dequantize(k_low, 'W_K')
        v_low = self.fp8_manager_v.dequantize(v_low, 'W_V')

        # Replace the corresponding selected head in the low-precision branch with the high-precision result.
        # Here, q_low, k_low, and v_low all have shape [batch, n_heads, pos, d_head].
        q_low[:, selected_head:selected_head+1] = q_high
        k_low[:, selected_head:selected_head+1] = k_high
        v_low[:, selected_head:selected_head+1] = v_high

        # Add the respective biases and pass through downstream hooks,
        # maintaining the tensor shape [batch, n_heads, pos, d_head].
        v_out = self.hook_v(v_low + self.b_V)
        q_out = self.hook_q(q_low + self.b_Q)
        k_out = self.hook_k(k_low + self.b_K)

        # Attention computation uses tensors in the [batch, n_heads, pos, d_head] order.
        # Compute attention scores:
        # Einsum equation: "batch,n_heads,pos,d_head, batch,n_heads,pos,d_head -> batch,n_heads,pos,pos"
        attn_scores = torch.einsum("batch,n_heads,pos,d_head, batch,n_heads,pos,d_head-> batch,n_heads,pos,pos",
                                    q_out, k_out) / self.attn_scale

        if past_kv_cache_entry is not None:
            # Append the new keys and values to the cache and update the cache position.
            kv_cache_pos_offset = past_kv_cache_entry.past_keys.size(1)
            k_out, v_out = past_kv_cache_entry.append(k_out, v_out)
        else:
            kv_cache_pos_offset = 0

        if self.cfg.positional_embedding_type == "rotary":
            q_out, k_out = self.rotary_rotate_qk(q_out, k_out, kv_cache_pos_offset)

        if self.cfg.attention_dir == "causal":
            attn_scores = self.apply_causal_mask(attn_scores, kv_cache_pos_offset, left_attention_mask)
        if additive_attention_mask is not None:
            attn_scores += additive_attention_mask

        attn_scores = self.hook_attn_scores(attn_scores)
        # Use softmax to obtain the attention distribution, with shape [batch, n_heads, pos, pos].
        pattern = self.hook_pattern(F.softmax(attn_scores, dim=-1))
        pattern = pattern.to(self.cfg.dtype)

        # Weighted sum: use einsum to compute the weighted combination.
        # Einsum equation: "batch,n_heads,pos_query,pos_key, batch,n_heads,pos_key,d_head -> batch,n_heads,pos_query,d_head"
        z = self.hook_z(torch.einsum("batch,n_heads,pos,pos, batch,n_heads,pos,d_head-> batch,n_heads,pos,d_head",
                                     pattern, v_out))

        # Subsequent O projection:
        if not self.cfg.use_attn_result:
            # Project each head individually:
            # Einsum equation: "batch,n_heads,pos,d_head, n_heads,d_head,d_model -> batch,n_heads,pos,d_model"
            o_proj = torch.einsum("batch,n_heads,pos,d_head, n_heads,d_head,d_model-> batch,n_heads,pos,d_model",
                                  z, self.W_O)
            # Sum over the n_heads dimension and add the bias to obtain the final output of shape [batch, pos, d_model].
            out = o_proj.sum(dim=1) + self.b_O
        else:
            result = self.hook_result(torch.einsum("batch,n_heads,pos,d_head, n_heads,d_head,d_model-> batch,n_heads,pos,d_model",
                                                   z, self.W_O))
            out = einops.reduce(result, "batch n_heads pos d_model -> batch pos d_model", "sum") + self.b_O

        return out
