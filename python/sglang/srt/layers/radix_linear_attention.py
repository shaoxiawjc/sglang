# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Radix linear attention."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.compilation.piecewise_context_manager import get_forward_context
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


# ---------------------------------------------------------------------------
# RRMC segment-level helpers
# ---------------------------------------------------------------------------

_FB_SEGMENT_FIELDS = (
    "extend_seq_lens",
    "extend_seq_lens_cpu",
    "extend_start_loc",
    "extend_prefix_lens",
    "mamba_track_mask",
    "mamba_track_seqlens",
    "mamba_track_indices",
    "extend_num_tokens",
)


def _save_fb_state(forward_batch) -> Dict[str, Any]:
    return {f: getattr(forward_batch, f, None) for f in _FB_SEGMENT_FIELDS}


def _restore_fb_state(forward_batch, saved: Dict[str, Any]) -> None:
    for f, v in saved.items():
        setattr(forward_batch, f, v)


def _setup_segment_fb(
    forward_batch, seg_idx: int, seg_len: int, is_last: bool, orig: Dict[str, Any]
) -> None:
    device = forward_batch.extend_seq_lens.device

    forward_batch.extend_num_tokens = seg_len
    forward_batch.extend_seq_lens = torch.tensor(
        [seg_len], dtype=orig["extend_seq_lens"].dtype, device=device
    )
    forward_batch.extend_seq_lens_cpu = [seg_len]
    forward_batch.extend_start_loc = torch.tensor(
        [0], dtype=orig["extend_start_loc"].dtype, device=device
    )

    if seg_idx == 0:
        forward_batch.extend_prefix_lens = orig["extend_prefix_lens"]
    else:
        forward_batch.extend_prefix_lens = torch.tensor(
            [1], dtype=orig["extend_prefix_lens"].dtype, device=device
        )

    if is_last:
        forward_batch.mamba_track_mask = orig["mamba_track_mask"]
        forward_batch.mamba_track_indices = orig["mamba_track_indices"]
        # mamba_track_seqlens must match the per-segment view:
        # seq_len = extend_seq_lens + extend_prefix_lens = seg_len + prefix_lens
        if orig["mamba_track_seqlens"] is not None:
            if seg_idx == 0:
                forward_batch.mamba_track_seqlens = orig["mamba_track_seqlens"]
            else:
                forward_batch.mamba_track_seqlens = torch.tensor(
                    [seg_len + forward_batch.extend_prefix_lens.item()],
                    dtype=orig["mamba_track_seqlens"].dtype,
                    device=device,
                )
    else:
        if orig["mamba_track_mask"] is not None:
            forward_batch.mamba_track_mask = torch.zeros_like(orig["mamba_track_mask"])
        if orig["mamba_track_seqlens"] is not None:
            forward_batch.mamba_track_seqlens = torch.full_like(
                orig["mamba_track_seqlens"], -1
            )


def _reinit_linear_attn_metadata(forward_batch) -> None:
    attn_backend = forward_batch.attn_backend
    linear_backend = getattr(attn_backend, "linear_attn_backend", attn_backend)
    linear_backend.init_forward_metadata(forward_batch)


class RadixLinearAttention(nn.Module):
    """
    The Linear Attention Layer Implementation.
    """

    # Set by model builder to indicate the total number of linear-attn layers.
    total_linear_layers: int = 0

    def __init__(
        self,
        layer_id: int,
        num_q_heads: int,
        num_k_heads: int,
        num_v_heads: int,
        head_q_dim: int,
        head_k_dim: int,
        head_v_dim: int,
        # GDN KDA Shared Weights
        conv_weights: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
        bias: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
        activation: str = "silu",
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.num_q_heads = num_q_heads
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_q_dim = head_q_dim
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.q_dim = num_q_heads * head_q_dim
        self.k_dim = num_k_heads * head_k_dim
        self.v_dim = num_v_heads * head_v_dim

        self.conv_weights = conv_weights
        self.bias = bias
        self.activation = activation

        self.A_log = A_log
        self.dt_bias = dt_bias

    def forward(
        self,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        segs = getattr(forward_batch, "rrmc_segments", None)
        is_rrmc = segs and len(segs) > 1

        if not is_rrmc:
            return self._forward_once(forward_batch, mixed_qkv, a, b)

        # Multi-segment RRMC path: process per-document sequentially,
        # reusing mamba state across segments via in-place pool updates.
        seg_outputs = []
        orig_state = _save_fb_state(forward_batch)
        last_linear_layer_id = self._get_last_linear_layer_id(forward_batch)

        for seg_idx, seg in enumerate(segs):
            is_last_seg = (seg_idx == len(segs) - 1)
            seg_len = seg.end - seg.start

            seg_mixed_qkv = mixed_qkv[seg.start:seg.end]
            seg_a = a[seg.start:seg.end]
            seg_b = b[seg.start:seg.end]

            _setup_segment_fb(
                forward_batch, seg_idx, seg_len, is_last_seg, orig_state
            )
            _reinit_linear_attn_metadata(forward_batch)

            seg_output = self._forward_once(
                forward_batch, seg_mixed_qkv, seg_a, seg_b
            )
            seg_outputs.append(seg_output)

            if not is_last_seg and self.layer_id == last_linear_layer_id:
                tree_cache = getattr(forward_batch, "tree_cache", None)
                if tree_cache is not None:
                    cache_seg = getattr(tree_cache, "cache_segment_state", None)
                    if callable(cache_seg):
                        # Pass only segments up to the current one so state is
                        # saved for the segment that just completed.
                        completed_segs = segs[: seg_idx + 1]
                        cache_seg(
                            forward_batch.rrmc_req,
                            completed_segs,
                            forward_batch.req_to_token_pool,
                        )

        _restore_fb_state(forward_batch, orig_state)
        return torch.cat(seg_outputs, dim=1)

    def _forward_once(
        self,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        if (
            forward_batch.forward_mode.is_extend()
            and get_forward_context() is not None
        ):
            seq_len = mixed_qkv.shape[0]
            output = torch.empty(
                (1, seq_len, self.num_v_heads, self.head_v_dim),
                dtype=mixed_qkv.dtype,
                device=mixed_qkv.device,
            )
            unified_linear_attention_with_output(
                mixed_qkv,
                a,
                b,
                output,
                self.layer_id,
            )
            return output
        else:
            return forward_batch.attn_backend.forward(
                layer=self,
                forward_batch=forward_batch,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
            )

    def _get_last_linear_layer_id(self, forward_batch) -> int:
        context = get_forward_context()
        if context is not None and context.attention_layers:
            return max(layer.layer_id for layer in context.attention_layers)
        fb_val = getattr(forward_batch, "rrmc_last_linear_layer_id", -1)
        if fb_val >= 0:
            return fb_val
        return self.layer_id


@register_custom_op(mutates_args=["output"])
@register_split_op()
def unified_linear_attention_with_output(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    layer_id: int,
) -> None:
    """
    Custom op wrapper for linear attention computation only.
    """
    context = get_forward_context()
    forward_batch = context.forward_batch
    attention_layers = context.attention_layers
    attention_layer = attention_layers[layer_id]

    ret = forward_batch.attn_backend.forward(
        layer=attention_layer,
        forward_batch=forward_batch,
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
    )

    assert (
        output.numel() == ret.numel()
    ), f"Output tensor element mismatch: {output.numel()} != {ret.numel()}"

    output.view(ret.shape).copy_(ret)
