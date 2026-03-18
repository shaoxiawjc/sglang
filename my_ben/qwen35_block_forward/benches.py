from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from einops import rearrange

from my_ben.qwen35_hybrid_recovery.utils import (
    add_row,
    benchmark_cuda_op,
    format_gb,
    gbps,
    nbytes_of_shape,
)
from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.server_args import ServerArgs

from .shapes import full_block_shape_info, linear_block_shape_info


def benchmark_cuda_op_with_setup(
    fn,
    *,
    warmup_iters: int,
    bench_iters: int,
    setup_fn=None,
):
    if setup_fn is None:
        return benchmark_cuda_op(fn, warmup_iters=warmup_iters, bench_iters=bench_iters)

    for _ in range(warmup_iters):
        setup_fn()
        torch.cuda.synchronize()
        fn()
    torch.cuda.synchronize()

    times_ms: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(bench_iters):
        setup_fn()
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        end.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    import numpy as np

    def percentile(values: list[float], q: float) -> float:
        return float(np.percentile(np.asarray(values, dtype=np.float64), q))

    return {
        "mean_ms": float(np.mean(times_ms)),
        "median_ms": float(np.median(times_ms)),
        "p20_ms": percentile(times_ms, 20),
        "p80_ms": percentile(times_ms, 80),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
    }


class LocalLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, *, bias: bool = False):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.linear(x), None


class LocalMergedLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: list[int], *, bias: bool = False):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, sum(out_features), bias=bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.linear(x), None


class LocalQwen2MoeMLP(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, *, bias: bool = False):
        super().__init__()
        self.gate_up_proj = torch.nn.Linear(
            hidden_size, intermediate_size * 2, bias=bias
        )
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        del should_allreduce_fusion, use_reduce_scatter
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class LocalQwen35LinearSubmodule(torch.nn.Module):
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_id: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.in_proj_qkv = LocalMergedLinear(
            self.hidden_size,
            [self.key_dim, self.key_dim, self.value_dim],
            bias=False,
        )
        self.in_proj_z = LocalLinear(
            self.hidden_size,
            self.value_dim,
            bias=False,
        )
        self.in_proj_b = LocalLinear(
            self.hidden_size,
            self.num_v_heads,
            bias=False,
        )
        self.in_proj_a = LocalLinear(
            self.hidden_size,
            self.num_v_heads,
            bias=False,
        )
        conv_weights = torch.randn(
            self.conv_dim,
            config.linear_conv_kernel_dim,
            dtype=dtype,
            device="cuda",
        )
        self.attn = RadixLinearAttention(
            layer_id=layer_id,
            num_q_heads=self.num_k_heads,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_weights=conv_weights,
            bias=None,
            activation=config.hidden_act,
            A_log=torch.randn(self.num_v_heads, dtype=torch.float32, device="cuda"),
            dt_bias=torch.ones(self.num_v_heads, dtype=torch.float32, device="cuda"),
        )
        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=config.rms_norm_eps,
            group_size=None,
            norm_before_gate=True,
            device=torch.get_device_module().current_device(),
            dtype=dtype,
        )
        self.out_proj = LocalLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        linear_backend: GDNAttnBackend,
    ) -> torch.Tensor:
        mixed_qkv, _ = self.in_proj_qkv(hidden_states)
        z, _ = self.in_proj_z(hidden_states)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b, _ = self.in_proj_b(hidden_states)
        a, _ = self.in_proj_a(hidden_states)
        core_attn_out = linear_backend.forward_extend(
            self.attn,
            forward_batch,
            mixed_qkv,
            a,
            b,
        )
        z_shape = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output, _ = self.out_proj(core_attn_out)
        return output


class LocalQwen35LinearBlock(torch.nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_id: int, dtype: torch.dtype):
        super().__init__()
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.linear_attn = LocalQwen35LinearSubmodule(config, layer_id, dtype)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = LocalQwen2MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.linear_attn(
            hidden_states,
            forward_batch,
            forward_batch.attn_backend,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states, False, False)
        return hidden_states, residual


class LocalQwen35FullBlock(torch.nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_id: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_heads = self.total_num_heads
        self.num_kv_heads = self.total_num_kv_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        rope_scaling = (
            getattr(config, "rope_parameters", None)
            if hasattr(config, "rope_parameters")
            else getattr(config, "rope_scaling", None)
        )
        rope_theta = rope_scaling.get("rope_theta", 10000)
        partial_rotary_factor = rope_scaling.get("partial_rotary_factor", 1.0)
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            rope_scaling=rope_scaling,
            base=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        qkv_out = self.q_size * (1 + self.attn_output_gate) + 2 * self.kv_size
        self.qkv_proj = LocalLinear(
            config.hidden_size,
            qkv_out,
            bias=False,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix="bench.full_block.attn",
        )
        self.o_proj = LocalLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = LocalQwen2MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if self.attn_output_gate:
            q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            gate = None
        q_by_head = self.q_norm(q.reshape(-1, self.head_dim)).view(q.shape)
        k_by_head = self.k_norm(k.reshape(-1, self.head_dim)).view(k.shape)
        q, k = self.rotary_emb(positions, q_by_head, k_by_head)
        attn_output = self.attn(q, k, v, forward_batch)
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attention(positions, hidden_states, forward_batch)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states, False, False)
        return hidden_states, residual


class LinearBlockBench:
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        dtype: torch.dtype,
        cache_params: Mamba2CacheParams,
        layer_id: int,
        max_batch_size: int,
        max_context_len: int,
    ):
        self.config = config
        self.dtype = dtype
        self.layer_id = layer_id
        self.req_pool = HybridReqToTokenPool(
            size=max_batch_size,
            mamba_size=max_batch_size,
            mamba_spec_state_size=max_batch_size,
            max_context_len=max_context_len,
            device="cuda",
            enable_memory_saver=False,
            cache_params=cache_params,
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=None,
        )
        self.req_pool.req_index_to_mamba_index_mapping[:max_batch_size] = torch.arange(
            1, max_batch_size + 1, dtype=torch.int32, device="cuda"
        )
        model_runner = SimpleNamespace(device="cuda", req_to_token_pool=self.req_pool)
        self.backend = GDNAttnBackend(model_runner)
        self.block = LocalQwen35LinearBlock(config, layer_id, dtype).to(
            device="cuda", dtype=dtype
        )
        self.cache_params = cache_params
        self.host_conv = [
            torch.randn(
                (max_batch_size + 1,) + conv_shape,
                dtype=cache_params.dtype.conv,
                device="cpu",
                pin_memory=True,
            )
            for conv_shape in cache_params.shape.conv
        ]
        self.host_temporal = torch.randn(
            (max_batch_size + 1,) + cache_params.shape.temporal,
            dtype=cache_params.dtype.temporal,
            device="cpu",
            pin_memory=True,
        )

    def make_forward_batch(
        self, batch_size: int, seq_len: int, prefix_len: int
    ) -> ForwardBatch:
        total_len = prefix_len + seq_len
        req_pool_indices = torch.arange(batch_size, device="cuda")
        return ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=batch_size,
            input_ids=torch.randint(
                0, 1024, (batch_size, seq_len), device="cuda", dtype=torch.int32
            ),
            req_pool_indices=req_pool_indices,
            seq_lens=torch.full((batch_size,), total_len, dtype=torch.int32, device="cuda"),
            out_cache_loc=torch.empty(batch_size * seq_len, dtype=torch.int32, device="cuda"),
            seq_lens_sum=batch_size * total_len,
            seq_lens_cpu=torch.full((batch_size,), total_len, dtype=torch.int32),
            extend_num_tokens=batch_size * seq_len,
            extend_seq_lens=torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda"),
            extend_prefix_lens=torch.full(
                (batch_size,), prefix_len, dtype=torch.int32, device="cuda"
            ),
            extend_start_loc=torch.arange(
                0, batch_size * seq_len, step=seq_len, dtype=torch.int32, device="cuda"
            ),
            extend_prefix_lens_cpu=torch.full((batch_size,), prefix_len, dtype=torch.int32),
            extend_seq_lens_cpu=torch.full((batch_size,), seq_len, dtype=torch.int32),
            req_to_token_pool=self.req_pool,
            attn_backend=self.backend,
        )

    def make_hidden_states(self, token_count: int) -> torch.Tensor:
        return torch.randn(token_count, self.config.hidden_size, dtype=self.dtype, device="cuda")

    def _slot_slice(self, batch_size: int) -> slice:
        return slice(1, batch_size + 1)

    def zero_state_slots(self, batch_size: int) -> None:
        layer_cache = self.req_pool.mamba2_layer_cache(self.layer_id)
        for conv_states in layer_cache.conv:
            conv_states[self._slot_slice(batch_size)].zero_()
        layer_cache.temporal[self._slot_slice(batch_size)].zero_()

    def load_state_to_gpu(self, batch_size: int) -> None:
        slot_slice = self._slot_slice(batch_size)
        layer_cache = self.req_pool.mamba2_layer_cache(self.layer_id)
        for conv_idx, conv_states in enumerate(layer_cache.conv):
            conv_states[slot_slice].copy_(
                self.host_conv[conv_idx][slot_slice],
                non_blocking=True,
            )
        layer_cache.temporal[slot_slice].copy_(
            self.host_temporal[slot_slice],
            non_blocking=True,
        )

    def bench_full_forward(
        self,
        batch_size: int,
        seq_len: int,
        prefix_len: int,
        *,
        warmup_iters: int,
        bench_iters: int,
    ) -> dict[str, float]:
        total_len = prefix_len + seq_len
        hidden_states = self.make_hidden_states(batch_size * total_len)
        forward_batch = self.make_forward_batch(batch_size, total_len, 0)
        self.backend.init_forward_metadata(forward_batch)

        def run():
            self.block(hidden_states, None, forward_batch)

        return benchmark_cuda_op_with_setup(
            run,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
            setup_fn=lambda: self.zero_state_slots(batch_size),
        )

    def bench_reuse_cache_compute(
        self,
        batch_size: int,
        seq_len: int,
        prefix_len: int,
        *,
        warmup_iters: int,
        bench_iters: int,
    ) -> dict[str, float]:
        hidden_states = self.make_hidden_states(batch_size * seq_len)
        forward_batch = self.make_forward_batch(batch_size, seq_len, prefix_len)
        self.backend.init_forward_metadata(forward_batch)

        def run():
            self.block(hidden_states, None, forward_batch)

        setup = (
            (lambda: self.zero_state_slots(batch_size))
            if prefix_len == 0
            else (lambda: self.load_state_to_gpu(batch_size))
        )
        return benchmark_cuda_op_with_setup(
            run,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
            setup_fn=setup,
        )

    def bench_state_load(
        self,
        batch_size: int,
        prefix_len: int,
        *,
        warmup_iters: int,
        bench_iters: int,
    ) -> dict[str, float]:
        if prefix_len == 0:
            return {
                "mean_ms": 0.0,
                "median_ms": 0.0,
                "p20_ms": 0.0,
                "p80_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
            }

        return benchmark_cuda_op_with_setup(
            lambda: self.load_state_to_gpu(batch_size),
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
            setup_fn=lambda: self.zero_state_slots(batch_size),
        )

    def state_bytes(self, batch_size: int, prefix_len: int) -> int:
        if prefix_len == 0:
            return 0
        return int(self.cache_params.mamba_cache_per_req * batch_size)


class FullBlockBench:
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        dtype: torch.dtype,
        layer_id: int,
        max_batch_size: int,
        max_total_len: int,
    ):
        self.config = config
        self.dtype = dtype
        self.layer_id = layer_id
        self.page_size = 1
        max_total_tokens = max_batch_size * max_total_len + self.page_size
        self.model_runner = SimpleNamespace(
            device="cuda",
            dtype=dtype,
            kv_cache_dtype=dtype,
            is_hybrid_swa=False,
            attention_chunk_size=None,
            sliding_window_size=None,
            page_size=self.page_size,
            token_to_kv_pool=MHATokenToKVPool(
                size=max_total_tokens,
                page_size=self.page_size,
                dtype=dtype,
                head_num=config.num_key_value_heads,
                head_dim=config.head_dim,
                layer_num=1,
                device="cuda",
                enable_memory_saver=False,
                start_layer=layer_id,
                end_layer=layer_id + 1,
            ),
            req_to_token_pool=SimpleNamespace(
                size=max_batch_size,
                req_to_token=torch.zeros(
                    max_batch_size, max_total_len, dtype=torch.int32, device="cuda"
                ),
            ),
            model_config=SimpleNamespace(
                context_len=max_total_len,
                is_multimodal=False,
                attention_arch=AttentionArch.MHA,
                is_encoder_decoder=False,
                is_local_attention_model=False,
                num_attention_heads=config.num_attention_heads,
            ),
            server_args=ServerArgs(model_path="dummy"),
        )
        self.backend = FlashAttentionBackend(self.model_runner)
        self.block = LocalQwen35FullBlock(config, layer_id).to(
            device="cuda", dtype=dtype
        )
        self.host_k = torch.randn(
            max_batch_size * max_total_len,
            config.num_key_value_heads,
            config.head_dim,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        self.host_v = torch.randn_like(self.host_k)
        self.stage_k = torch.empty(
            max_batch_size * max_total_len,
            config.num_key_value_heads,
            config.head_dim,
            dtype=dtype,
            device="cuda",
        )
        self.stage_v = torch.empty_like(self.stage_k)

    def make_forward_batch(
        self, batch_size: int, seq_len: int, prefix_len: int
    ) -> ForwardBatch:
        total_len = prefix_len + seq_len
        req_pool_indices = torch.arange(batch_size, device="cuda")
        token_map = (
            torch.arange(batch_size, device="cuda", dtype=torch.int32)[:, None] * total_len
            + torch.arange(total_len, device="cuda", dtype=torch.int32)[None, :]
            + self.page_size
        )
        self.model_runner.req_to_token_pool.req_to_token[:batch_size, :total_len] = token_map
        out_cache_loc = token_map[:, prefix_len:].reshape(-1).contiguous()
        return ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=batch_size,
            input_ids=torch.randint(
                0, 1024, (batch_size, seq_len), device="cuda", dtype=torch.int32
            ),
            req_pool_indices=req_pool_indices,
            seq_lens=torch.full((batch_size,), total_len, dtype=torch.int32, device="cuda"),
            out_cache_loc=out_cache_loc,
            seq_lens_sum=batch_size * total_len,
            seq_lens_cpu=torch.full((batch_size,), total_len, dtype=torch.int32),
            extend_num_tokens=batch_size * seq_len,
            extend_seq_lens=torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda"),
            extend_prefix_lens=torch.full(
                (batch_size,), prefix_len, dtype=torch.int32, device="cuda"
            ),
            extend_start_loc=torch.arange(
                0, batch_size * seq_len, step=seq_len, dtype=torch.int32, device="cuda"
            ),
            extend_prefix_lens_cpu=torch.full((batch_size,), prefix_len, dtype=torch.int32),
            extend_seq_lens_cpu=torch.full((batch_size,), seq_len, dtype=torch.int32),
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.backend,
        )

    def make_hidden_states(self, token_count: int) -> torch.Tensor:
        return torch.randn(token_count, self.config.hidden_size, dtype=self.dtype, device="cuda")

    def make_positions(self, batch_size: int, seq_len: int, prefix_len: int) -> torch.Tensor:
        base = torch.arange(prefix_len, prefix_len + seq_len, device="cuda", dtype=torch.int64)
        return base.repeat(batch_size)

    def clear_prefix_kv(self) -> None:
        self.model_runner.token_to_kv_pool.k_buffer[0].zero_()
        self.model_runner.token_to_kv_pool.v_buffer[0].zero_()

    def load_prefix_kv(self, forward_batch: ForwardBatch, prefix_len: int) -> None:
        if prefix_len == 0:
            return
        batch_size = forward_batch.batch_size
        token_count = batch_size * prefix_len
        loc = (
            self.model_runner.req_to_token_pool.req_to_token[:batch_size, :prefix_len]
            .reshape(-1)
            .contiguous()
        )
        self.stage_k[:token_count].copy_(self.host_k[:token_count], non_blocking=True)
        self.stage_v[:token_count].copy_(self.host_v[:token_count], non_blocking=True)
        self.model_runner.token_to_kv_pool.set_kv_buffer(
            self.block.attn,
            loc,
            self.stage_k[:token_count],
            self.stage_v[:token_count],
            self.block.attn.k_scale,
            self.block.attn.v_scale,
        )

    def bench_full_forward(
        self,
        batch_size: int,
        seq_len: int,
        prefix_len: int,
        *,
        warmup_iters: int,
        bench_iters: int,
    ) -> dict[str, float]:
        total_len = prefix_len + seq_len
        hidden_states = self.make_hidden_states(batch_size * total_len)
        positions = self.make_positions(batch_size, total_len, 0)
        forward_batch = self.make_forward_batch(batch_size, total_len, 0)
        self.backend.init_forward_metadata(forward_batch)

        def run():
            self.block(positions, hidden_states, None, forward_batch)

        return benchmark_cuda_op_with_setup(
            run,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
            setup_fn=self.clear_prefix_kv,
        )

    def bench_reuse_cache_compute(
        self,
        batch_size: int,
        seq_len: int,
        prefix_len: int,
        *,
        warmup_iters: int,
        bench_iters: int,
    ) -> dict[str, float]:
        hidden_states = self.make_hidden_states(batch_size * seq_len)
        positions = self.make_positions(batch_size, seq_len, prefix_len)
        forward_batch = self.make_forward_batch(batch_size, seq_len, prefix_len)
        self.backend.init_forward_metadata(forward_batch)

        def run():
            self.block(positions, hidden_states, None, forward_batch)

        setup = (
            self.clear_prefix_kv
            if prefix_len == 0
            else lambda: self.load_prefix_kv(forward_batch, prefix_len)
        )
        return benchmark_cuda_op_with_setup(
            run,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
            setup_fn=setup,
        )

    def bench_kv_load(
        self,
        batch_size: int,
        seq_len: int,
        prefix_len: int,
        *,
        warmup_iters: int,
        bench_iters: int,
    ) -> dict[str, float]:
        if prefix_len == 0:
            return {
                "mean_ms": 0.0,
                "median_ms": 0.0,
                "p20_ms": 0.0,
                "p80_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
            }

        forward_batch = self.make_forward_batch(batch_size, seq_len, prefix_len)
        return benchmark_cuda_op_with_setup(
            lambda: self.load_prefix_kv(forward_batch, prefix_len),
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
            setup_fn=self.clear_prefix_kv,
        )

    def kv_bytes(self, batch_size: int, prefix_len: int) -> int:
        if prefix_len == 0:
            return 0
        token_count = batch_size * prefix_len
        k_bytes = nbytes_of_shape(
            (token_count, self.config.num_key_value_heads, self.config.head_dim),
            self.dtype,
        )
        v_bytes = nbytes_of_shape(
            (token_count, self.config.num_key_value_heads, self.config.head_dim),
            self.dtype,
        )
        return int(k_bytes + v_bytes)


def run_block_forward_benchmarks(
    rows: list[dict[str, object]],
    *,
    args: argparse.Namespace,
    config: Qwen3_5TextConfig,
    model_dtype: torch.dtype,
    cache_params: Mamba2CacheParams,
    linear_layer_id: int,
    full_layer_id: int,
) -> None:
    linear_bench = LinearBlockBench(
        config,
        model_dtype,
        cache_params,
        linear_layer_id,
        max_batch_size=max(args.batch_sizes),
        max_context_len=max(args.prefix_lens) + max(args.seq_lens),
    )
    full_bench = FullBlockBench(
        config,
        model_dtype,
        full_layer_id,
        max_batch_size=max(args.batch_sizes),
        max_total_len=max(args.prefix_lens) + max(args.seq_lens),
    )
    for batch_size in args.batch_sizes:
        for seq_len in args.seq_lens:
            for prefix_len in args.prefix_lens:
                total_len = prefix_len + seq_len
                linear_full_forward_stats = linear_bench.bench_full_forward(
                    batch_size,
                    seq_len,
                    prefix_len,
                    warmup_iters=args.warmup_iters,
                    bench_iters=args.bench_iters,
                )
                linear_reuse_stats = linear_bench.bench_reuse_cache_compute(
                    batch_size,
                    seq_len,
                    prefix_len,
                    warmup_iters=args.warmup_iters,
                    bench_iters=args.bench_iters,
                )
                linear_state_load_stats = linear_bench.bench_state_load(
                    batch_size,
                    prefix_len,
                    warmup_iters=args.warmup_iters,
                    bench_iters=args.bench_iters,
                )
                full_full_forward_stats = full_bench.bench_full_forward(
                    batch_size,
                    seq_len,
                    prefix_len,
                    warmup_iters=args.warmup_iters,
                    bench_iters=args.bench_iters,
                )
                full_reuse_stats = full_bench.bench_reuse_cache_compute(
                    batch_size,
                    seq_len,
                    prefix_len,
                    warmup_iters=args.warmup_iters,
                    bench_iters=args.bench_iters,
                )
                full_kv_load_stats = full_bench.bench_kv_load(
                    batch_size,
                    seq_len,
                    prefix_len,
                    warmup_iters=args.warmup_iters,
                    bench_iters=args.bench_iters,
                )

                linear_shape = linear_block_shape_info(
                    config,
                    cache_params,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    prefix_len=prefix_len,
                    total_len=total_len,
                )
                full_shape = full_block_shape_info(
                    config,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    prefix_len=prefix_len,
                    total_len=total_len,
                )
                payload = {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "prefix_len": prefix_len,
                    "total_len": total_len,
                }
                linear_state_bytes = linear_bench.state_bytes(batch_size, prefix_len)
                full_kv_bytes = full_bench.kv_bytes(batch_size, prefix_len)
                add_row(
                    rows,
                    category="linear_block_full_forward",
                    stats=linear_full_forward_stats,
                    payload={**payload, "layer_id": linear_layer_id, "shape_info": linear_shape},
                )
                add_row(
                    rows,
                    category="linear_block_reuse_cache_compute",
                    stats=linear_reuse_stats,
                    payload={**payload, "layer_id": linear_layer_id, "shape_info": linear_shape},
                )
                add_row(
                    rows,
                    category="linear_state_cache_h2d",
                    stats=linear_state_load_stats,
                    payload={
                        **payload,
                        "layer_id": linear_layer_id,
                        "shape_info": linear_shape,
                        "bytes": linear_state_bytes,
                        "bytes_gb": format_gb(linear_state_bytes),
                        "gbps": gbps(linear_state_bytes, linear_state_load_stats["median_ms"])
                        if linear_state_load_stats["median_ms"] > 0
                        else 0.0,
                    },
                )
                add_row(
                    rows,
                    category="full_block_full_forward",
                    stats=full_full_forward_stats,
                    payload={**payload, "layer_id": full_layer_id, "shape_info": full_shape},
                )
                add_row(
                    rows,
                    category="full_block_reuse_cache_compute",
                    stats=full_reuse_stats,
                    payload={**payload, "layer_id": full_layer_id, "shape_info": full_shape},
                )
                add_row(
                    rows,
                    category="full_kvcache_h2d",
                    stats=full_kv_load_stats,
                    payload={
                        **payload,
                        "layer_id": full_layer_id,
                        "shape_info": full_shape,
                        "bytes": full_kv_bytes,
                        "bytes_gb": format_gb(full_kv_bytes),
                        "gbps": gbps(full_kv_bytes, full_kv_load_stats["median_ms"])
                        if full_kv_load_stats["median_ms"] > 0
                        else 0.0,
                    },
                )
