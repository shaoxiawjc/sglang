from __future__ import annotations

import argparse
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from my_ben.qwen35_block_forward.benches import (
    LocalQwen35FullBlock,
    LocalQwen35LinearBlock,
    benchmark_cuda_op_with_setup,
)
from my_ben.qwen35_hybrid_recovery.utils import (
    add_row,
    format_gb,
    gbps,
    nbytes_of_shape,
)
from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.server_args import ServerArgs

from .shapes import strategy_shape_info


@dataclass
class LinearBlockContext:
    forward_batch: ForwardBatch
    hidden_states: torch.Tensor


@dataclass
class CausalBlockContext:
    batch_size: int
    seq_len: int
    prefix_len: int
    token_map: torch.Tensor
    forward_batch: ForwardBatch
    positions: torch.Tensor
    hidden_states: torch.Tensor

    @property
    def total_len(self) -> int:
        return self.seq_len + self.prefix_len


class LinearBlockGroupBench:
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        dtype: torch.dtype,
        cache_params: Mamba2CacheParams,
        layer_ids: list[int],
        max_batch_size: int,
        max_context_len: int,
    ):
        self.config = config
        self.dtype = dtype
        self.layer_ids = layer_ids
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
        self.blocks = [
            LocalQwen35LinearBlock(config, layer_id, dtype).to(
                device="cuda", dtype=dtype
            )
            for layer_id in layer_ids
        ]
        self.cache_params = cache_params
        self.host_conv = [
            torch.randn(
                (len(layer_ids), max_batch_size + 1) + conv_shape,
                dtype=cache_params.dtype.conv,
                device="cpu",
                pin_memory=True,
            )
            for conv_shape in cache_params.shape.conv
        ]
        self.host_temporal = torch.randn(
            (len(layer_ids), max_batch_size + 1) + cache_params.shape.temporal,
            dtype=cache_params.dtype.temporal,
            device="cpu",
            pin_memory=True,
        )
        self.conv_bytes_per_slot_per_layer = sum(
            nbytes_of_shape(conv_shape, cache_params.dtype.conv)
            for conv_shape in cache_params.shape.conv
        )
        self.temporal_bytes_per_slot_per_layer = nbytes_of_shape(
            cache_params.shape.temporal,
            cache_params.dtype.temporal,
        )
        self.state_bytes_per_slot_per_layer = (
            self.conv_bytes_per_slot_per_layer + self.temporal_bytes_per_slot_per_layer
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

    def make_context(
        self, batch_size: int, seq_len: int, prefix_len: int
    ) -> LinearBlockContext:
        return LinearBlockContext(
            forward_batch=self.make_forward_batch(batch_size, seq_len, prefix_len),
            hidden_states=self.make_hidden_states(batch_size * seq_len),
        )

    def _slot_slice(self, batch_size: int) -> slice:
        return slice(1, batch_size + 1)

    def zero_all_states(self, batch_size: int) -> None:
        slot_slice = self._slot_slice(batch_size)
        for layer_id in self.layer_ids:
            layer_cache = self.req_pool.mamba2_layer_cache(layer_id)
            for conv_states in layer_cache.conv:
                conv_states[slot_slice].zero_()
            layer_cache.temporal[slot_slice].zero_()

    def prefetch_conv(self, layer_positions: list[int], batch_size: int) -> None:
        if not layer_positions:
            return
        slot_slice = self._slot_slice(batch_size)
        for position in layer_positions:
            layer_cache = self.req_pool.mamba2_layer_cache(self.layer_ids[position])
            for conv_idx, conv_states in enumerate(layer_cache.conv):
                conv_states[slot_slice].copy_(
                    self.host_conv[conv_idx][position, slot_slice],
                    non_blocking=True,
                )
            layer_cache.temporal[slot_slice].copy_(
                self.host_temporal[position, slot_slice],
                non_blocking=True,
            )

    def run_context(
        self,
        layer_positions: list[int],
        context: LinearBlockContext,
    ) -> None:
        if not layer_positions:
            return
        self.prepare_context(context)
        hidden_states = context.hidden_states
        residual = None
        for position in layer_positions:
            hidden_states, residual = self.run_one_layer(
                position,
                hidden_states,
                residual,
                context,
            )

    def conv_bytes(self, batch_size: int, layer_count: int) -> int:
        return int(batch_size * layer_count * self.conv_bytes_per_slot_per_layer)

    def state_bytes(self, batch_size: int, layer_count: int) -> int:
        return int(batch_size * layer_count * self.state_bytes_per_slot_per_layer)

    def prepare_context(self, context: LinearBlockContext) -> None:
        self.backend.init_forward_metadata(context.forward_batch)

    def run_one_layer(
        self,
        position: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        context: LinearBlockContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.blocks[position](
            hidden_states,
            residual,
            context.forward_batch,
        )


class CausalBlockBench:
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
        self.bytes_per_prefix_token = 2 * nbytes_of_shape(
            (1, config.num_key_value_heads, config.head_dim),
            dtype,
        )

    def make_positions(self, batch_size: int, seq_len: int, prefix_len: int) -> torch.Tensor:
        base = torch.arange(prefix_len, prefix_len + seq_len, device="cuda", dtype=torch.int64)
        return base.repeat(batch_size)

    def make_hidden_states(self, token_count: int) -> torch.Tensor:
        return torch.randn(token_count, self.config.hidden_size, dtype=self.dtype, device="cuda")

    def make_context(
        self, batch_size: int, seq_len: int, prefix_len: int
    ) -> CausalBlockContext:
        total_len = prefix_len + seq_len
        token_map = (
            torch.arange(batch_size, device="cuda", dtype=torch.int32)[:, None] * total_len
            + torch.arange(total_len, device="cuda", dtype=torch.int32)[None, :]
            + self.page_size
        )
        out_cache_loc = token_map[:, prefix_len:].reshape(-1).contiguous()
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=batch_size,
            input_ids=torch.randint(
                0, 1024, (batch_size, seq_len), device="cuda", dtype=torch.int32
            ),
            req_pool_indices=torch.arange(batch_size, device="cuda"),
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
        return CausalBlockContext(
            batch_size=batch_size,
            seq_len=seq_len,
            prefix_len=prefix_len,
            token_map=token_map,
            forward_batch=forward_batch,
            positions=self.make_positions(batch_size, seq_len, prefix_len),
            hidden_states=self.make_hidden_states(batch_size * seq_len),
        )

    def activate_context(self, context: CausalBlockContext) -> None:
        self.model_runner.req_to_token_pool.req_to_token[
            : context.batch_size, : context.total_len
        ] = context.token_map

    def clear_prefix_kv(self) -> None:
        self.model_runner.token_to_kv_pool.k_buffer[0].zero_()
        self.model_runner.token_to_kv_pool.v_buffer[0].zero_()

    def prefetch_prefix_kv(self, context: CausalBlockContext) -> None:
        if context.prefix_len == 0:
            return
        self.activate_context(context)
        token_count = context.batch_size * context.prefix_len
        loc = context.token_map[:, : context.prefix_len].reshape(-1).contiguous()
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

    def run_context(self, context: CausalBlockContext) -> None:
        self.activate_context(context)
        self.backend.init_forward_metadata(context.forward_batch)
        self.block(
            context.positions,
            context.hidden_states,
            None,
            context.forward_batch,
        )

    def kv_bytes(self, batch_size: int, prefix_len: int) -> int:
        return int(batch_size * prefix_len * self.bytes_per_prefix_token)


def _reset_group(
    linear_bench: LinearBlockGroupBench,
    causal_bench: CausalBlockBench,
    batch_size: int,
) -> None:
    linear_bench.zero_all_states(batch_size)
    causal_bench.clear_prefix_kv()


def _run_baseline_recompute(
    *,
    linear_bench: LinearBlockGroupBench,
    linear_total_context: LinearBlockContext,
    causal_bench: CausalBlockBench,
    causal_total_context: CausalBlockContext,
    linear_positions: list[int],
) -> None:
    linear_bench.run_context(linear_positions, linear_total_context)
    causal_bench.run_context(causal_total_context)


def _run_baseline_offload_onload(
    *,
    linear_bench: LinearBlockGroupBench,
    linear_current_context: LinearBlockContext,
    causal_bench: CausalBlockBench,
    causal_current_context: CausalBlockContext,
    linear_positions: list[int],
) -> None:
    linear_bench.prefetch_conv(linear_positions, linear_current_context.forward_batch.batch_size)
    causal_bench.prefetch_prefix_kv(causal_current_context)
    linear_bench.run_context(linear_positions, linear_current_context)
    causal_bench.run_context(causal_current_context)


def _run_ours_ca_first(
    *,
    linear_bench: LinearBlockGroupBench,
    linear_current_context: LinearBlockContext,
    causal_bench: CausalBlockBench,
    causal_total_context: CausalBlockContext,
    linear_positions: list[int],
    recompute_stream: torch.cuda.Stream,
    prefetch_stream: torch.cuda.Stream,
) -> None:
    compute_done = torch.cuda.Event()
    prefetch_events: list[torch.cuda.Event] = []

    with torch.cuda.stream(recompute_stream):
        causal_bench.run_context(causal_total_context)
        compute_done.record()

    with torch.cuda.stream(prefetch_stream):
        for position in linear_positions:
            linear_bench.prefetch_conv(
                [position], linear_current_context.forward_batch.batch_size
            )
            prefetch_done = torch.cuda.Event()
            prefetch_done.record()
            prefetch_events.append(prefetch_done)

    current_stream = torch.cuda.current_stream()
    current_stream.wait_event(compute_done)
    linear_bench.prepare_context(linear_current_context)
    hidden_states = linear_current_context.hidden_states
    residual = None
    for position, prefetch_done in zip(linear_positions, prefetch_events):
        current_stream.wait_event(prefetch_done)
        hidden_states, residual = linear_bench.run_one_layer(
            position,
            hidden_states,
            residual,
            linear_current_context,
        )


def _run_ours_la_first(
    *,
    linear_bench: LinearBlockGroupBench,
    linear_total_context: LinearBlockContext,
    linear_current_context: LinearBlockContext,
    causal_bench: CausalBlockBench,
    causal_current_context: CausalBlockContext,
    recompute_positions: list[int],
    onload_positions: list[int],
    recompute_stream: torch.cuda.Stream,
    prefetch_stream: torch.cuda.Stream,
) -> None:
    copy_done = torch.cuda.Event()
    compute_done = torch.cuda.Event()

    with torch.cuda.stream(recompute_stream):
        linear_bench.run_context(recompute_positions, linear_total_context)
        compute_done.record()

    with torch.cuda.stream(prefetch_stream):
        causal_bench.prefetch_prefix_kv(causal_current_context)
        copy_done.record()

    current_stream = torch.cuda.current_stream()
    current_stream.wait_event(copy_done)
    current_stream.wait_event(compute_done)

    if onload_positions:
        prefetch_events: list[torch.cuda.Event] = []
        with torch.cuda.stream(prefetch_stream):
            for position in onload_positions:
                linear_bench.prefetch_conv(
                    [position], linear_current_context.forward_batch.batch_size
                )
                prefetch_done = torch.cuda.Event()
                prefetch_done.record()
                prefetch_events.append(prefetch_done)
        linear_bench.prepare_context(linear_current_context)
        hidden_states = linear_current_context.hidden_states
        residual = None
        for position, prefetch_done in zip(onload_positions, prefetch_events):
            current_stream.wait_event(prefetch_done)
            hidden_states, residual = linear_bench.run_one_layer(
                position,
                hidden_states,
                residual,
                linear_current_context,
            )
    causal_bench.run_context(causal_current_context)


def run_overlap_benchmarks(
    rows: list[dict[str, object]],
    *,
    args: argparse.Namespace,
    config: Qwen3_5TextConfig,
    model_dtype: torch.dtype,
    linear_cache_params: Mamba2CacheParams,
    linear_layer_ids: list[int],
    causal_layer_id: int,
    recompute_counts: list[int],
) -> None:
    linear_bench = LinearBlockGroupBench(
        config,
        model_dtype,
        linear_cache_params,
        linear_layer_ids,
        max_batch_size=max(args.batch_sizes),
        max_context_len=max(args.prefix_lens) + max(args.seq_lens),
    )
    causal_bench = CausalBlockBench(
        config,
        model_dtype,
        causal_layer_id,
        max_batch_size=max(args.batch_sizes),
        max_total_len=max(args.prefix_lens) + max(args.seq_lens),
    )
    recompute_stream = torch.cuda.Stream()
    prefetch_stream = torch.cuda.Stream()
    linear_positions = list(range(len(linear_layer_ids)))

    for batch_size in args.batch_sizes:
        for seq_len in args.seq_lens:
            for prefix_len in args.prefix_lens:
                total_len = prefix_len + seq_len
                linear_total_context = linear_bench.make_context(batch_size, total_len, 0)
                linear_current_context = linear_bench.make_context(
                    batch_size, seq_len, prefix_len
                )
                causal_total_context = causal_bench.make_context(batch_size, total_len, 0)
                causal_current_context = causal_bench.make_context(
                    batch_size, seq_len, prefix_len
                )

                baseline_recompute_stats = benchmark_cuda_op_with_setup(
                    lambda: _run_baseline_recompute(
                        linear_bench=linear_bench,
                        linear_total_context=linear_total_context,
                        causal_bench=causal_bench,
                        causal_total_context=causal_total_context,
                        linear_positions=linear_positions,
                    ),
                    warmup_iters=args.warmup_iters,
                    bench_iters=args.bench_iters,
                    setup_fn=lambda: _reset_group(linear_bench, causal_bench, batch_size),
                )
                baseline_offload_stats = benchmark_cuda_op_with_setup(
                    lambda: _run_baseline_offload_onload(
                        linear_bench=linear_bench,
                        linear_current_context=linear_current_context,
                        causal_bench=causal_bench,
                        causal_current_context=causal_current_context,
                        linear_positions=linear_positions,
                    ),
                    warmup_iters=args.warmup_iters,
                    bench_iters=args.bench_iters,
                    setup_fn=lambda: _reset_group(linear_bench, causal_bench, batch_size),
                )
                ours_ca_first_stats = benchmark_cuda_op_with_setup(
                    lambda: _run_ours_ca_first(
                        linear_bench=linear_bench,
                        linear_current_context=linear_current_context,
                        causal_bench=causal_bench,
                        causal_total_context=causal_total_context,
                        linear_positions=linear_positions,
                        recompute_stream=recompute_stream,
                        prefetch_stream=prefetch_stream,
                    ),
                    warmup_iters=args.warmup_iters,
                    bench_iters=args.bench_iters,
                    setup_fn=lambda: _reset_group(linear_bench, causal_bench, batch_size),
                )

                baseline_linear_bytes = linear_bench.state_bytes(
                    batch_size, len(linear_positions)
                )
                baseline_causal_bytes = causal_bench.kv_bytes(batch_size, prefix_len)
                base_payload = {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "prefix_len": prefix_len,
                    "total_len": total_len,
                    "linear_layer_ids": linear_layer_ids,
                    "causal_layer_id": causal_layer_id,
                }
                add_row(
                    rows,
                    category="baseline_recompute",
                    stats=baseline_recompute_stats,
                    payload={
                        **base_payload,
                        "strategy_family": "baseline",
                        "strategy_order": "la_then_ca",
                        "linear_recompute_count": len(linear_positions),
                        "linear_onload_count": 0,
                        "bytes": 0,
                        "bytes_gb": format_gb(0),
                        "gbps": 0.0,
                        "shape_info": strategy_shape_info(
                            config=config,
                            cache_params=linear_cache_params,
                            batch_size=batch_size,
                            seq_len=seq_len,
                            prefix_len=prefix_len,
                            linear_layer_count=len(linear_positions),
                            linear_recompute_count=len(linear_positions),
                        ),
                    },
                )
                baseline_total_bytes = baseline_linear_bytes + baseline_causal_bytes
                add_row(
                    rows,
                    category="baseline_offload_onload",
                    stats=baseline_offload_stats,
                    payload={
                        **base_payload,
                        "strategy_family": "baseline",
                        "strategy_order": "la_then_ca",
                        "linear_recompute_count": 0,
                        "linear_onload_count": len(linear_positions),
                        "bytes": baseline_total_bytes,
                        "bytes_gb": format_gb(baseline_total_bytes),
                        "gbps": gbps(baseline_total_bytes, baseline_offload_stats["median_ms"])
                        if baseline_total_bytes > 0
                        else 0.0,
                        "shape_info": strategy_shape_info(
                            config=config,
                            cache_params=linear_cache_params,
                            batch_size=batch_size,
                            seq_len=seq_len,
                            prefix_len=prefix_len,
                            linear_layer_count=len(linear_positions),
                            linear_recompute_count=0,
                        ),
                    },
                )

                ours_ca_bytes = baseline_linear_bytes
                add_row(
                    rows,
                    category="ours_ca_recompute_overlap_la_state_conv",
                    stats=ours_ca_first_stats,
                    payload={
                        **base_payload,
                        "strategy_family": "ours",
                        "strategy_order": "ca_then_la",
                        "linear_recompute_count": 0,
                        "linear_onload_count": len(linear_positions),
                        "causal_recompute_count": 1,
                        "bytes": ours_ca_bytes,
                        "bytes_gb": format_gb(ours_ca_bytes),
                        "gbps": gbps(ours_ca_bytes, ours_ca_first_stats["median_ms"])
                        if ours_ca_bytes > 0
                        else 0.0,
                        "shape_info": strategy_shape_info(
                            config=config,
                            cache_params=linear_cache_params,
                            batch_size=batch_size,
                            seq_len=seq_len,
                            prefix_len=prefix_len,
                            linear_layer_count=len(linear_positions),
                            linear_recompute_count=0,
                        ),
                    },
                )

                for recompute_count in recompute_counts:
                    recompute_positions = linear_positions[:recompute_count]
                    onload_positions = linear_positions[recompute_count:]
                    ours_la_stats = benchmark_cuda_op_with_setup(
                        lambda rp=recompute_positions, op=onload_positions: _run_ours_la_first(
                            linear_bench=linear_bench,
                            linear_total_context=linear_total_context,
                            linear_current_context=linear_current_context,
                            causal_bench=causal_bench,
                            causal_current_context=causal_current_context,
                            recompute_positions=rp,
                            onload_positions=op,
                            recompute_stream=recompute_stream,
                            prefetch_stream=prefetch_stream,
                        ),
                        warmup_iters=args.warmup_iters,
                        bench_iters=args.bench_iters,
                        setup_fn=lambda: _reset_group(linear_bench, causal_bench, batch_size),
                    )
                    overlap_bytes = baseline_causal_bytes
                    total_bytes = overlap_bytes + linear_bench.state_bytes(
                        batch_size, len(onload_positions)
                    )
                    add_row(
                        rows,
                        category="ours_la_recompute_overlap_ca_kvcache",
                        stats=ours_la_stats,
                        payload={
                            **base_payload,
                            "strategy_family": "ours",
                            "strategy_order": "la_then_ca",
                            "linear_recompute_count": recompute_count,
                            "linear_onload_count": len(onload_positions),
                            "causal_recompute_count": 0,
                            "bytes": total_bytes,
                            "bytes_gb": format_gb(total_bytes),
                            "overlap_bytes": overlap_bytes,
                            "overlap_bytes_gb": format_gb(overlap_bytes),
                            "gbps": gbps(total_bytes, ours_la_stats["median_ms"])
                            if total_bytes > 0
                            else 0.0,
                            "shape_info": strategy_shape_info(
                                config=config,
                                cache_params=linear_cache_params,
                                batch_size=batch_size,
                                seq_len=seq_len,
                                prefix_len=prefix_len,
                                linear_layer_count=len(linear_positions),
                                linear_recompute_count=recompute_count,
                            ),
                        },
                    )
