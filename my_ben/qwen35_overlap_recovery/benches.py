from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.server_args import ServerArgs

from my_ben.qwen35_hybrid_recovery.utils import (
    add_row,
    benchmark_cuda_op,
    format_gb,
    gbps,
    nbytes_of_shape,
)

from .shapes import overlap_shape_info


class LinearRecoveryBench:
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        dtype: torch.dtype,
        cache_params: Mamba2CacheParams,
        linear_layer_ids: list[int],
        max_batch_size: int,
        max_context_len: int,
    ):
        self.config = config
        self.dtype = dtype
        self.linear_layer_ids = linear_layer_ids
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
        self.layers = [self._build_layer(layer_id) for layer_id in linear_layer_ids]

    def _build_layer(self, layer_id: int) -> RadixLinearAttention:
        conv_dim = (
            self.config.linear_num_key_heads * self.config.linear_key_head_dim * 2
            + self.config.linear_num_value_heads * self.config.linear_value_head_dim
        )
        return RadixLinearAttention(
            layer_id=int(layer_id),
            num_q_heads=self.config.linear_num_key_heads,
            num_k_heads=self.config.linear_num_key_heads,
            num_v_heads=self.config.linear_num_value_heads,
            head_q_dim=self.config.linear_key_head_dim,
            head_k_dim=self.config.linear_key_head_dim,
            head_v_dim=self.config.linear_value_head_dim,
            conv_weights=torch.randn(
                conv_dim,
                self.config.linear_conv_kernel_dim,
                dtype=self.dtype,
                device="cuda",
            ),
            bias=None,
            activation=self.config.hidden_act,
            A_log=torch.randn(
                self.config.linear_num_value_heads,
                dtype=torch.float32,
                device="cuda",
            ),
            dt_bias=torch.ones(
                self.config.linear_num_value_heads,
                dtype=torch.float32,
                device="cuda",
            ),
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
            seq_lens=torch.full(
                (batch_size,), total_len, dtype=torch.int32, device="cuda"
            ),
            out_cache_loc=torch.empty(
                batch_size * seq_len, dtype=torch.int32, device="cuda"
            ),
            seq_lens_sum=batch_size * total_len,
            seq_lens_cpu=torch.full((batch_size,), total_len, dtype=torch.int32),
            extend_num_tokens=batch_size * seq_len,
            extend_seq_lens=torch.full(
                (batch_size,), seq_len, dtype=torch.int32, device="cuda"
            ),
            extend_prefix_lens=torch.full(
                (batch_size,), prefix_len, dtype=torch.int32, device="cuda"
            ),
            extend_start_loc=torch.arange(
                0, batch_size * seq_len, step=seq_len, dtype=torch.int32, device="cuda"
            ),
            extend_prefix_lens_cpu=torch.full(
                (batch_size,), prefix_len, dtype=torch.int32
            ),
            extend_seq_lens_cpu=torch.full((batch_size,), seq_len, dtype=torch.int32),
            req_to_token_pool=self.req_pool,
            attn_backend=self.backend,
        )

    def make_inputs(self, batch_size: int, seq_len: int) -> list[tuple[torch.Tensor, ...]]:
        total_tokens = batch_size * seq_len
        inputs: list[tuple[torch.Tensor, ...]] = []
        for layer in self.layers:
            mixed_qkv = torch.randn(
                total_tokens,
                layer.q_dim + layer.k_dim + layer.v_dim,
                dtype=self.dtype,
                device="cuda",
            )
            a = torch.randn(
                total_tokens,
                self.config.linear_num_value_heads,
                dtype=self.dtype,
                device="cuda",
            )
            b = torch.randn_like(a)
            inputs.append((mixed_qkv, a, b))
        return inputs

    def run_layers(
        self,
        forward_batch: ForwardBatch,
        inputs: list[tuple[torch.Tensor, ...]],
        layer_positions: list[int],
    ) -> None:
        if not layer_positions:
            return
        self.backend.init_forward_metadata(forward_batch)
        for position in layer_positions:
            layer = self.layers[position]
            mixed_qkv, a, b = inputs[position]
            self.backend.forward_extend(layer, forward_batch, mixed_qkv, a, b)


class LinearStatePrefetchBench:
    def __init__(self, linear_bench: LinearRecoveryBench, cache_params: Mamba2CacheParams):
        self.linear_bench = linear_bench
        self.cache_params = cache_params
        self.num_layers = len(cache_params.layers)
        self.host_conv = [
            torch.randn(
                (self.num_layers,) + conv_shape,
                dtype=cache_params.dtype.conv,
                device="cpu",
                pin_memory=True,
            )
            for conv_shape in cache_params.shape.conv
        ]
        self.host_temporal = torch.randn(
            (self.num_layers,) + cache_params.shape.temporal,
            dtype=cache_params.dtype.temporal,
            device="cpu",
            pin_memory=True,
        )
        self.bytes_per_slot_per_layer = sum(
            nbytes_of_shape(conv_shape, cache_params.dtype.conv)
            for conv_shape in cache_params.shape.conv
        ) + nbytes_of_shape(cache_params.shape.temporal, cache_params.dtype.temporal)

    def prefetch(self, layer_positions: list[int], req_pool_indices: torch.Tensor) -> None:
        if not layer_positions:
            return
        cache_indices = self.linear_bench.req_pool.get_mamba_indices(req_pool_indices)
        batch_size = int(cache_indices.shape[0])
        for position in layer_positions:
            layer_id = self.linear_bench.linear_layer_ids[position]
            layer_cache = self.linear_bench.req_pool.mamba2_layer_cache(layer_id)
            for conv_idx, conv_states in enumerate(layer_cache.conv):
                host_conv = self.host_conv[conv_idx][position].unsqueeze(0).expand(
                    batch_size, *self.host_conv[conv_idx][position].shape
                )
                conv_states[cache_indices].copy_(host_conv, non_blocking=True)
            host_temporal = self.host_temporal[position].unsqueeze(0).expand(
                batch_size, *self.host_temporal[position].shape
            )
            layer_cache.temporal[cache_indices].copy_(
                host_temporal, non_blocking=True
            )


class FullAttentionBench:
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        dtype: torch.dtype,
        full_layer_ids: list[int],
        max_batch_size: int,
        max_total_len: int,
    ):
        self.config = config
        self.dtype = dtype
        self.full_layer_ids = full_layer_ids
        self.page_size = 1
        max_total_num_tokens = max_batch_size * max_total_len + self.page_size
        self.model_runner = SimpleNamespace(
            device="cuda",
            dtype=dtype,
            kv_cache_dtype=dtype,
            is_hybrid_swa=False,
            attention_chunk_size=None,
            sliding_window_size=None,
            page_size=self.page_size,
            token_to_kv_pool=MHATokenToKVPool(
                size=max_total_num_tokens,
                page_size=self.page_size,
                dtype=dtype,
                head_num=config.num_key_value_heads,
                head_dim=config.head_dim,
                layer_num=len(full_layer_ids),
                device="cuda",
                enable_memory_saver=False,
            ),
            req_to_token_pool=SimpleNamespace(
                size=max_batch_size,
                req_to_token=torch.zeros(
                    max_batch_size,
                    max_total_len,
                    dtype=torch.int32,
                    device="cuda",
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
        self.layers = [
            RadixAttention(
                num_heads=config.num_attention_heads,
                head_dim=config.head_dim,
                scaling=config.head_dim**-0.5,
                num_kv_heads=config.num_key_value_heads,
                layer_id=layer_pos,
            )
            for layer_pos in range(len(full_layer_ids))
        ]
        self.max_kv_tokens = max_batch_size * max_total_len
        self.host_k = [
            torch.randn(
                self.max_kv_tokens,
                config.num_key_value_heads,
                config.head_dim,
                dtype=dtype,
                device="cpu",
                pin_memory=True,
            )
            for _ in full_layer_ids
        ]
        self.host_v = [torch.randn_like(k) for k in self.host_k]
        self.q_proj_weights = [
            torch.randn(
                config.num_attention_heads * config.head_dim,
                config.hidden_size,
                dtype=dtype,
                device="cuda",
            )
            for _ in full_layer_ids
        ]
        self.k_proj_weights = [
            torch.randn(
                config.num_key_value_heads * config.head_dim,
                config.hidden_size,
                dtype=dtype,
                device="cuda",
            )
            for _ in full_layer_ids
        ]
        self.v_proj_weights = [
            torch.randn(
                config.num_key_value_heads * config.head_dim,
                config.hidden_size,
                dtype=dtype,
                device="cuda",
            )
            for _ in full_layer_ids
        ]
        self.bytes_per_prefix_kv_token_per_layer = (
            config.num_key_value_heads * config.head_dim * dtype.itemsize * 2
        )

    def make_forward_batch(
        self, batch_size: int, seq_len: int, prefix_len: int
    ) -> ForwardBatch:
        total_len = prefix_len + seq_len
        req_pool_indices = torch.arange(batch_size, device="cuda")
        token_map = (
            torch.arange(batch_size, device="cuda", dtype=torch.int32)[:, None]
            * total_len
            + torch.arange(total_len, device="cuda", dtype=torch.int32)[None, :]
            + self.page_size
        )
        self.model_runner.req_to_token_pool.req_to_token[:batch_size, :total_len] = (
            token_map
        )
        out_cache_loc = token_map[:, prefix_len:].reshape(-1).contiguous()
        return ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=batch_size,
            input_ids=torch.randint(
                0, 1024, (batch_size, seq_len), device="cuda", dtype=torch.int32
            ),
            req_pool_indices=req_pool_indices,
            seq_lens=torch.full(
                (batch_size,), total_len, dtype=torch.int32, device="cuda"
            ),
            out_cache_loc=out_cache_loc,
            seq_lens_sum=batch_size * total_len,
            seq_lens_cpu=torch.full((batch_size,), total_len, dtype=torch.int32),
            extend_num_tokens=batch_size * seq_len,
            extend_seq_lens=torch.full(
                (batch_size,), seq_len, dtype=torch.int32, device="cuda"
            ),
            extend_prefix_lens=torch.full(
                (batch_size,), prefix_len, dtype=torch.int32, device="cuda"
            ),
            extend_start_loc=torch.arange(
                0, batch_size * seq_len, step=seq_len, dtype=torch.int32, device="cuda"
            ),
            extend_prefix_lens_cpu=torch.full(
                (batch_size,), prefix_len, dtype=torch.int32
            ),
            extend_seq_lens_cpu=torch.full((batch_size,), seq_len, dtype=torch.int32),
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.backend,
        )

    def prefetch_prefix_kv(self, forward_batch: ForwardBatch, prefix_len: int) -> None:
        if prefix_len == 0:
            return
        batch_size = forward_batch.batch_size
        token_count = batch_size * prefix_len
        loc = (
            self.model_runner.req_to_token_pool.req_to_token[:batch_size, :prefix_len]
            .reshape(-1)
            .contiguous()
        )
        for layer_index, layer in enumerate(self.layers):
            cache_k = self.host_k[layer_index][:token_count].to(
                "cuda", non_blocking=True
            )
            cache_v = self.host_v[layer_index][:token_count].to(
                "cuda", non_blocking=True
            )
            self.model_runner.token_to_kv_pool.set_kv_buffer(
                layer,
                loc,
                cache_k,
                cache_v,
                layer.k_scale,
                layer.v_scale,
            )

    def make_hidden_states(self, batch_size: int, seq_len: int) -> torch.Tensor:
        total_tokens = batch_size * seq_len
        return torch.randn(
            total_tokens,
            self.config.hidden_size,
            dtype=self.dtype,
            device="cuda",
        )

    def _qkv_projection(
        self, hidden_states: torch.Tensor, layer_index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = F.linear(hidden_states, self.q_proj_weights[layer_index]).view(
            -1, self.config.num_attention_heads, self.config.head_dim
        )
        k = F.linear(hidden_states, self.k_proj_weights[layer_index]).view(
            -1, self.config.num_key_value_heads, self.config.head_dim
        )
        v = F.linear(hidden_states, self.v_proj_weights[layer_index]).view(
            -1, self.config.num_key_value_heads, self.config.head_dim
        )
        return q, k, v

    def run_layers(self, forward_batch: ForwardBatch, hidden_states: torch.Tensor) -> None:
        self.backend.init_forward_metadata(forward_batch)
        for layer_index, layer in enumerate(self.layers):
            q, k, v = self._qkv_projection(hidden_states, layer_index)
            self.backend.forward_extend(q, k, v, layer, forward_batch)


def run_overlap_benchmarks(
    rows: list[dict[str, object]],
    *,
    args: argparse.Namespace,
    config: Qwen3_5TextConfig,
    model_dtype: torch.dtype,
    linear_cache_params: Mamba2CacheParams,
    target_linear_layer_ids: list[int],
    target_full_layer_ids: list[int],
    recompute_counts: list[int],
) -> None:
    linear_bench = LinearRecoveryBench(
        config,
        model_dtype,
        linear_cache_params,
        target_linear_layer_ids,
        max_batch_size=max(args.batch_sizes),
        max_context_len=max(args.prefix_lens) + max(args.seq_lens),
    )
    state_bench = LinearStatePrefetchBench(linear_bench, linear_cache_params)
    full_bench = FullAttentionBench(
        config,
        model_dtype,
        target_full_layer_ids,
        max_batch_size=max(args.batch_sizes),
        max_total_len=max(args.prefix_lens) + max(args.seq_lens),
    )
    recompute_stream = torch.cuda.Stream()
    prefetch_stream = torch.cuda.Stream()

    total_linear_positions = list(range(len(target_linear_layer_ids)))

    for batch_size in args.batch_sizes:
        for seq_len in args.seq_lens:
            for prefix_len in args.prefix_lens:
                total_len = prefix_len + seq_len
                linear_current_batch = linear_bench.make_forward_batch(
                    batch_size, seq_len, prefix_len
                )
                linear_recompute_batch = linear_bench.make_forward_batch(
                    batch_size, total_len, 0
                )
                linear_current_inputs = linear_bench.make_inputs(batch_size, seq_len)
                linear_recompute_inputs = linear_bench.make_inputs(batch_size, total_len)

                full_current_batch = full_bench.make_forward_batch(
                    batch_size, seq_len, prefix_len
                )
                full_recompute_batch = full_bench.make_forward_batch(
                    batch_size, total_len, 0
                )
                full_current_hidden_states = full_bench.make_hidden_states(
                    batch_size, seq_len
                )
                full_recompute_hidden_states = full_bench.make_hidden_states(
                    batch_size, total_len
                )

                shape_info = overlap_shape_info(
                    config=config,
                    cache_params=linear_cache_params,
                    linear_layer_ids=target_linear_layer_ids,
                    full_layer_ids=target_full_layer_ids,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    prefix_len=prefix_len,
                )

                prefix_linear_bytes = (
                    batch_size
                    * state_bench.bytes_per_slot_per_layer
                    * len(target_linear_layer_ids)
                )
                prefix_full_kv_bytes = (
                    batch_size
                    * prefix_len
                    * full_bench.bytes_per_prefix_kv_token_per_layer
                    * len(target_full_layer_ids)
                )

                all_recompute_stats = benchmark_cuda_op(
                    lambda: _run_all_recompute_strategy(
                        linear_bench=linear_bench,
                        linear_recompute_batch=linear_recompute_batch,
                        linear_recompute_inputs=linear_recompute_inputs,
                        linear_positions=total_linear_positions,
                        full_bench=full_bench,
                        full_recompute_batch=full_recompute_batch,
                        full_recompute_hidden_states=full_recompute_hidden_states,
                    ),
                    warmup_iters=args.warmup_iters,
                    bench_iters=args.bench_iters,
                )
                all_onload_stats = benchmark_cuda_op(
                    lambda: _run_all_onload_strategy(
                        linear_bench=linear_bench,
                        state_bench=state_bench,
                        linear_current_batch=linear_current_batch,
                        linear_current_inputs=linear_current_inputs,
                        linear_positions=total_linear_positions,
                        full_bench=full_bench,
                        full_current_batch=full_current_batch,
                        full_current_hidden_states=full_current_hidden_states,
                        prefix_len=prefix_len,
                    ),
                    warmup_iters=args.warmup_iters,
                    bench_iters=args.bench_iters,
                )

                for recompute_count in recompute_counts:
                    recompute_positions = list(range(recompute_count))
                    onload_positions = list(
                        range(recompute_count, len(target_linear_layer_ids))
                    )
                    overlap_bytes = (
                        batch_size
                        * state_bench.bytes_per_slot_per_layer
                        * len(onload_positions)
                        + prefix_full_kv_bytes
                    )
                    overlap_stats = benchmark_cuda_op(
                        lambda: _run_overlap_strategy(
                            linear_bench=linear_bench,
                            state_bench=state_bench,
                            linear_recompute_batch=linear_recompute_batch,
                            linear_recompute_inputs=linear_recompute_inputs,
                            linear_current_batch=linear_current_batch,
                            linear_current_inputs=linear_current_inputs,
                            recompute_positions=recompute_positions,
                            onload_positions=onload_positions,
                            full_bench=full_bench,
                            full_current_batch=full_current_batch,
                            full_current_hidden_states=full_current_hidden_states,
                            prefix_len=prefix_len,
                            recompute_stream=recompute_stream,
                            prefetch_stream=prefetch_stream,
                        ),
                        warmup_iters=args.warmup_iters,
                        bench_iters=args.bench_iters,
                    )

                    common_payload = {
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "prefix_len": prefix_len,
                        "linear_recompute_count": recompute_count,
                        "linear_onload_count": len(onload_positions),
                        "linear_layer_ids": target_linear_layer_ids,
                        "full_layer_ids": target_full_layer_ids,
                        "shape_info": shape_info,
                    }

                    add_row(
                        rows,
                        category="strategy_all_recompute",
                        stats=all_recompute_stats,
                        payload=common_payload,
                    )
                    add_row(
                        rows,
                        category="strategy_overlap_linear_recompute_then_prefetch",
                        stats=overlap_stats,
                        payload={
                            **common_payload,
                            "bytes": overlap_bytes,
                            "bytes_gb": format_gb(overlap_bytes),
                            "gbps": gbps(overlap_bytes, overlap_stats["median_ms"])
                            if overlap_bytes > 0
                            else 0.0,
                        },
                    )
                    all_onload_bytes = prefix_linear_bytes + prefix_full_kv_bytes
                    add_row(
                        rows,
                        category="strategy_all_onload",
                        stats=all_onload_stats,
                        payload={
                            **common_payload,
                            "bytes": all_onload_bytes,
                            "bytes_gb": format_gb(all_onload_bytes),
                            "gbps": gbps(all_onload_bytes, all_onload_stats["median_ms"])
                            if all_onload_bytes > 0
                            else 0.0,
                        },
                    )


def _run_all_recompute_strategy(
    *,
    linear_bench: LinearRecoveryBench,
    linear_recompute_batch: ForwardBatch,
    linear_recompute_inputs: list[tuple[torch.Tensor, ...]],
    linear_positions: list[int],
    full_bench: FullAttentionBench,
    full_recompute_batch: ForwardBatch,
    full_recompute_hidden_states: torch.Tensor,
) -> None:
    linear_bench.run_layers(
        linear_recompute_batch, linear_recompute_inputs, linear_positions
    )
    full_bench.run_layers(full_recompute_batch, full_recompute_hidden_states)


def _run_all_onload_strategy(
    *,
    linear_bench: LinearRecoveryBench,
    state_bench: LinearStatePrefetchBench,
    linear_current_batch: ForwardBatch,
    linear_current_inputs: list[tuple[torch.Tensor, ...]],
    linear_positions: list[int],
    full_bench: FullAttentionBench,
    full_current_batch: ForwardBatch,
    full_current_hidden_states: torch.Tensor,
    prefix_len: int,
) -> None:
    state_bench.prefetch(linear_positions, linear_current_batch.req_pool_indices)
    full_bench.prefetch_prefix_kv(full_current_batch, prefix_len)
    linear_bench.run_layers(linear_current_batch, linear_current_inputs, linear_positions)
    full_bench.run_layers(full_current_batch, full_current_hidden_states)


def _run_overlap_strategy(
    *,
    linear_bench: LinearRecoveryBench,
    state_bench: LinearStatePrefetchBench,
    linear_recompute_batch: ForwardBatch,
    linear_recompute_inputs: list[tuple[torch.Tensor, ...]],
    linear_current_batch: ForwardBatch,
    linear_current_inputs: list[tuple[torch.Tensor, ...]],
    recompute_positions: list[int],
    onload_positions: list[int],
    full_bench: FullAttentionBench,
    full_current_batch: ForwardBatch,
    full_current_hidden_states: torch.Tensor,
    prefix_len: int,
    recompute_stream: torch.cuda.Stream,
    prefetch_stream: torch.cuda.Stream,
) -> None:
    copy_done = torch.cuda.Event()
    recompute_done = torch.cuda.Event()

    with torch.cuda.stream(prefetch_stream):
        state_bench.prefetch(onload_positions, linear_current_batch.req_pool_indices)
        full_bench.prefetch_prefix_kv(full_current_batch, prefix_len)
        copy_done.record()

    with torch.cuda.stream(recompute_stream):
        linear_bench.run_layers(
            linear_recompute_batch,
            linear_recompute_inputs,
            recompute_positions,
        )
        recompute_done.record()

    current_stream = torch.cuda.current_stream()
    current_stream.wait_event(copy_done)
    current_stream.wait_event(recompute_done)

    linear_bench.run_layers(linear_current_batch, linear_current_inputs, onload_positions)
    full_bench.run_layers(full_current_batch, full_current_hidden_states)
