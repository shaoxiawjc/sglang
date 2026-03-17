from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MHATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.server_args import ServerArgs

from .shapes import (
    full_attention_shape_info,
    kv_transfer_shape_info,
    linear_attention_shape_info,
    state_shape_info,
)
from .utils import add_row, benchmark_cuda_op, format_gb, gbps, nbytes_of_shape


class KVTransferTorchBench:
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        dtype: torch.dtype,
        max_tokens: int,
        full_layer_ids: list[int],
    ):
        self.full_layer_ids = full_layer_ids
        self.layer_num = len(self.full_layer_ids)
        self.head_num = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.host_kv = torch.randn(
            2,
            self.layer_num,
            max_tokens,
            self.head_num,
            self.head_dim,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        self.device_kv = torch.empty(
            2,
            self.layer_num,
            max_tokens,
            self.head_num,
            self.head_dim,
            dtype=dtype,
            device="cuda",
        )
        self.bytes_per_token_per_layer = (
            config.num_key_value_heads * config.head_dim * dtype.itemsize * 2
        )
        self.index_cache: dict[int, torch.Tensor] = {}

    def make_indices(self, token_count: int) -> torch.Tensor:
        if token_count not in self.index_cache:
            self.index_cache[token_count] = torch.arange(token_count, dtype=torch.int64)
        return self.index_cache[token_count]

    def bench_one_layer(
        self, token_count: int, layer_id: int, *, warmup_iters: int, bench_iters: int
    ) -> dict[str, float]:
        indices = self.make_indices(token_count)

        def fn() -> None:
            self.device_kv[:, layer_id, :token_count].copy_(
                self.host_kv[:, layer_id, indices], non_blocking=True
            )

        return benchmark_cuda_op(
            fn, warmup_iters=warmup_iters, bench_iters=bench_iters
        )


class KVTransferSGLangKernelBench:
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        dtype: torch.dtype,
        max_tokens: int,
        full_layer_ids: list[int],
    ):
        self.full_layer_ids = full_layer_ids
        self.layer_num = len(self.full_layer_ids)
        self.device_pool = MHATokenToKVPool(
            size=max_tokens + 1,
            page_size=1,
            dtype=dtype,
            head_num=config.num_key_value_heads,
            head_dim=config.head_dim,
            layer_num=self.layer_num,
            device="cuda",
            enable_memory_saver=False,
        )
        self.host_pool = MHATokenToKVPoolHost(
            device_pool=self.device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=1,
            layout="layer_first",
            pin_memory=True,
            device="cpu",
        )
        self.host_pool.kv_buffer.normal_()
        self.bytes_per_token_per_layer = (
            config.num_key_value_heads * config.head_dim * dtype.itemsize * 2
        )
        self.host_index_cache: dict[int, torch.Tensor] = {}
        self.device_index_cache: dict[int, torch.Tensor] = {}

    def make_host_indices(self, token_count: int) -> torch.Tensor:
        if token_count not in self.host_index_cache:
            self.host_index_cache[token_count] = torch.arange(
                token_count, dtype=torch.int64, device="cuda"
            )
        return self.host_index_cache[token_count]

    def make_device_indices(self, token_count: int) -> torch.Tensor:
        if token_count not in self.device_index_cache:
            self.device_index_cache[token_count] = torch.arange(
                1, token_count + 1, dtype=torch.int64, device="cuda"
            )
        return self.device_index_cache[token_count]

    def bench_one_layer(
        self, token_count: int, layer_id: int, *, warmup_iters: int, bench_iters: int
    ) -> dict[str, float]:
        host_indices = self.make_host_indices(token_count)
        device_indices = self.make_device_indices(token_count)

        def fn() -> None:
            self.host_pool.load_to_device_per_layer(
                self.device_pool,
                host_indices,
                device_indices,
                layer_id,
                "kernel",
            )

        return benchmark_cuda_op(
            fn, warmup_iters=warmup_iters, bench_iters=bench_iters
        )


class StateTransferBench:
    def __init__(
        self,
        cache_params: Mamba2CacheParams,
        max_slots: int,
        layer_count: int = 1,
    ):
        self.cache_params = cache_params
        self.num_layers = layer_count
        self.gpu_conv = [
            torch.empty(
                (self.num_layers, max_slots + 1) + conv_shape,
                dtype=cache_params.dtype.conv,
                device="cuda",
            )
            for conv_shape in cache_params.shape.conv
        ]
        self.gpu_temporal = torch.empty(
            (self.num_layers, max_slots + 1) + cache_params.shape.temporal,
            dtype=cache_params.dtype.temporal,
            device="cuda",
        )
        self.host_conv = [
            torch.randn(
                (self.num_layers, max_slots + 1) + conv_shape,
                dtype=cache_params.dtype.conv,
                device="cpu",
                pin_memory=True,
            )
            for conv_shape in cache_params.shape.conv
        ]
        self.host_temporal = torch.randn(
            (self.num_layers, max_slots + 1) + cache_params.shape.temporal,
            dtype=cache_params.dtype.temporal,
            device="cpu",
            pin_memory=True,
        )
        self.conv_bytes_per_slot = sum(
            nbytes_of_shape(conv_shape, cache_params.dtype.conv)
            for conv_shape in cache_params.shape.conv
        ) * self.num_layers
        self.temporal_bytes_per_slot = (
            nbytes_of_shape(cache_params.shape.temporal, cache_params.dtype.temporal)
            * self.num_layers
        )
        self.total_bytes_per_slot = (
            self.conv_bytes_per_slot + self.temporal_bytes_per_slot
        )
        self.index_cache: dict[int, torch.Tensor] = {}

    def make_indices(self, slot_count: int) -> torch.Tensor:
        if slot_count not in self.index_cache:
            self.index_cache[slot_count] = torch.arange(
                1, slot_count + 1, dtype=torch.int64
            )
        return self.index_cache[slot_count]

    def _copy_conv(self, slot_count: int) -> None:
        indices = self.make_indices(slot_count)
        slot_slice = slice(1, slot_count + 1)
        for dst, src in zip(self.gpu_conv, self.host_conv):
            dst[:, slot_slice].copy_(src[:, indices], non_blocking=True)

    def _copy_temporal(self, slot_count: int) -> None:
        indices = self.make_indices(slot_count)
        slot_slice = slice(1, slot_count + 1)
        self.gpu_temporal[:, slot_slice].copy_(
            self.host_temporal[:, indices], non_blocking=True
        )

    def bench(
        self,
        slot_count: int,
        component: str,
        *,
        warmup_iters: int,
        bench_iters: int,
    ) -> dict[str, float]:
        if component == "conv":
            fn = lambda: self._copy_conv(slot_count)
        elif component == "temporal":
            fn = lambda: self._copy_temporal(slot_count)
        elif component == "total":
            fn = lambda: (self._copy_conv(slot_count), self._copy_temporal(slot_count))
        else:
            raise ValueError(f"Unsupported component: {component}")
        return benchmark_cuda_op(
            fn, warmup_iters=warmup_iters, bench_iters=bench_iters
        )


class LinearAttentionBench:
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        dtype: torch.dtype,
        mamba_cache_params: Mamba2CacheParams,
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
            cache_params=mamba_cache_params,
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=None,
        )
        self.req_pool.req_index_to_mamba_index_mapping[:max_batch_size] = torch.arange(
            1, max_batch_size + 1, dtype=torch.int32, device="cuda"
        )
        model_runner = SimpleNamespace(device="cuda", req_to_token_pool=self.req_pool)
        self.backend = GDNAttnBackend(model_runner)
        layer_id = int(self.linear_layer_ids[0])
        conv_dim = (
            config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim
        )
        self.layer = RadixLinearAttention(
            layer_id=layer_id,
            num_q_heads=config.linear_num_key_heads,
            num_k_heads=config.linear_num_key_heads,
            num_v_heads=config.linear_num_value_heads,
            head_q_dim=config.linear_key_head_dim,
            head_k_dim=config.linear_key_head_dim,
            head_v_dim=config.linear_value_head_dim,
            conv_weights=torch.randn(
                conv_dim,
                config.linear_conv_kernel_dim,
                dtype=dtype,
                device="cuda",
            ),
            bias=None,
            activation=config.hidden_act,
            A_log=torch.randn(
                config.linear_num_value_heads, dtype=torch.float32, device="cuda"
            ),
            dt_bias=torch.ones(
                config.linear_num_value_heads, dtype=torch.float32, device="cuda"
            ),
        )

    def make_forward_batch(
        self, batch_size: int, seq_len: int
    ) -> ForwardBatch:
        total_len = seq_len
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
            extend_prefix_lens=torch.zeros(
                (batch_size,), dtype=torch.int32, device="cuda"
            ),
            extend_start_loc=torch.arange(
                0, batch_size * seq_len, step=seq_len, dtype=torch.int32, device="cuda"
            ),
            extend_prefix_lens_cpu=torch.zeros((batch_size,), dtype=torch.int32),
            extend_seq_lens_cpu=torch.full((batch_size,), seq_len, dtype=torch.int32),
            req_to_token_pool=self.req_pool,
            attn_backend=self.backend,
        )

    def bench(
        self,
        batch_size: int,
        seq_len: int,
        *,
        warmup_iters: int,
        bench_iters: int,
    ) -> dict[str, float]:
        forward_batch = self.make_forward_batch(batch_size, seq_len)
        self.backend.init_forward_metadata(forward_batch)

        total_tokens = batch_size * seq_len
        mixed_qkv = torch.randn(
            total_tokens,
            self.layer.q_dim + self.layer.k_dim + self.layer.v_dim,
            dtype=self.dtype,
            device="cuda",
        )
        a = torch.randn(
            total_tokens,
            self.config.linear_num_value_heads,
            dtype=self.dtype,
            device="cuda",
        )
        b = torch.randn(
            total_tokens,
            self.config.linear_num_value_heads,
            dtype=self.dtype,
            device="cuda",
        )

        def fn() -> None:
            self.backend.forward_extend(self.layer, forward_batch, mixed_qkv, a, b)

        return benchmark_cuda_op(
            fn, warmup_iters=warmup_iters, bench_iters=bench_iters
        )


class FullAttentionBench:
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        dtype: torch.dtype,
        max_batch_size: int,
        max_total_len: int,
    ):
        self.config = config
        self.dtype = dtype
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
                layer_num=1,
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
        self.layer = RadixAttention(
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            scaling=config.head_dim**-0.5,
            num_kv_heads=config.num_key_value_heads,
            layer_id=0,
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
        self.model_runner.req_to_token_pool.req_to_token[
            :batch_size, :total_len
        ] = token_map
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

    def _seed_prefix_kv(self, forward_batch: ForwardBatch, prefix_len: int) -> None:
        if prefix_len == 0:
            return
        batch_size = forward_batch.batch_size
        loc = (
            self.model_runner.req_to_token_pool.req_to_token[
                :batch_size, :prefix_len
            ].reshape(-1)
        )
        cache_k = torch.randn(
            batch_size * prefix_len,
            self.config.num_key_value_heads,
            self.config.head_dim,
            dtype=self.dtype,
            device="cuda",
        )
        cache_v = torch.randn_like(cache_k)
        self.model_runner.token_to_kv_pool.set_kv_buffer(
            self.layer,
            loc,
            cache_k,
            cache_v,
            self.layer.k_scale,
            self.layer.v_scale,
        )

    def bench(
        self,
        batch_size: int,
        seq_len: int,
        prefix_len: int,
        *,
        warmup_iters: int,
        bench_iters: int,
    ) -> dict[str, float]:
        forward_batch = self.make_forward_batch(batch_size, seq_len, prefix_len)
        self._seed_prefix_kv(forward_batch, prefix_len)
        self.backend.init_forward_metadata(forward_batch)

        total_tokens = batch_size * seq_len
        q = torch.randn(
            total_tokens,
            self.config.num_attention_heads,
            self.config.head_dim,
            dtype=self.dtype,
            device="cuda",
        )
        k = torch.randn(
            total_tokens,
            self.config.num_key_value_heads,
            self.config.head_dim,
            dtype=self.dtype,
            device="cuda",
        )
        v = torch.randn_like(k)

        def fn() -> None:
            self.backend.forward_extend(q, k, v, self.layer, forward_batch)

        return benchmark_cuda_op(
            fn, warmup_iters=warmup_iters, bench_iters=bench_iters
        )


def run_kv_transfer_benchmarks(
    rows: list[dict[str, object]],
    *,
    args: argparse.Namespace,
    config: Qwen3_5TextConfig,
    model_dtype: torch.dtype,
    full_layer_ids: list[int],
) -> None:
    torch_bench = KVTransferTorchBench(
        config, model_dtype, max(args.kv_token_counts), full_layer_ids
    )
    kernel_bench = KVTransferSGLangKernelBench(
        config, model_dtype, max(args.kv_token_counts), full_layer_ids
    )
    for token_count in args.kv_token_counts:
        one_layer_bytes = token_count * torch_bench.bytes_per_token_per_layer
        shared_payload = {
            "token_count": token_count,
            "full_layer_count": len(full_layer_ids),
            "bytes": one_layer_bytes,
            "bytes_gb": format_gb(one_layer_bytes),
            "shape_info": kv_transfer_shape_info(
                token_count,
                config.num_key_value_heads,
                config.head_dim,
                1,
            ),
        }
        for bench_name, category, bench in (
            (
                "torch_indexed",
                "kvcache_h2d_one_full_layer_torch_indexed",
                torch_bench,
            ),
            (
                "sglang_kernel",
                "kvcache_h2d_one_full_layer_sglang_kernel",
                kernel_bench,
            ),
        ):
            stats = bench.bench_one_layer(
                token_count,
                layer_id=0,
                warmup_iters=args.warmup_iters,
                bench_iters=args.bench_iters,
            )
            add_row(
                rows,
                category=category,
                stats=stats,
                payload={
                    **shared_payload,
                    "transfer_impl": bench_name,
                    "gbps": gbps(one_layer_bytes, stats["median_ms"]),
                },
            )


def run_state_transfer_benchmarks(
    rows: list[dict[str, object]],
    *,
    args: argparse.Namespace,
    cache_params: Mamba2CacheParams,
    linear_layer_ids: list[int],
) -> None:
    state_layer_count = 1
    state_bench = StateTransferBench(
        cache_params, max(args.state_slot_counts), layer_count=state_layer_count
    )
    for slot_count in args.state_slot_counts:
        shape_info = state_shape_info(cache_params, slot_count, state_layer_count)
        for component, bytes_per_slot in (
            ("conv", state_bench.conv_bytes_per_slot),
            ("temporal", state_bench.temporal_bytes_per_slot),
            ("total", state_bench.total_bytes_per_slot),
        ):
            stats = state_bench.bench(
                slot_count,
                component,
                warmup_iters=args.warmup_iters,
                bench_iters=args.bench_iters,
            )
            total_bytes = slot_count * bytes_per_slot
            add_row(
                rows,
                category=f"state_h2d_{component}",
                stats=stats,
                payload={
                    "slot_count": slot_count,
                    "linear_layer_count": state_layer_count,
                    "bytes": total_bytes,
                    "bytes_gb": format_gb(total_bytes),
                    "gbps": gbps(total_bytes, stats["median_ms"]),
                    "shape_info": shape_info,
                },
            )


def run_linear_attention_benchmarks(
    rows: list[dict[str, object]],
    *,
    args: argparse.Namespace,
    config: Qwen3_5TextConfig,
    model_dtype: torch.dtype,
    cache_params: Mamba2CacheParams,
    linear_layer_ids: list[int],
) -> None:
    linear_bench = LinearAttentionBench(
        config,
        model_dtype,
        cache_params,
        linear_layer_ids,
        max_batch_size=max(args.linear_batch_sizes),
        max_context_len=max(args.linear_seq_lens),
    )
    for batch_size in args.linear_batch_sizes:
        for seq_len in args.linear_seq_lens:
            stats = linear_bench.bench(
                batch_size,
                seq_len,
                warmup_iters=args.warmup_iters,
                bench_iters=args.bench_iters,
            )
            total_q_tokens = batch_size * seq_len
            add_row(
                rows,
                category="linear_attention_extend",
                stats=stats,
                payload={
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "q_tokens": total_q_tokens,
                    "tokens_per_s": total_q_tokens / (stats["median_ms"] / 1000.0),
                    "shape_info": linear_attention_shape_info(
                        linear_bench.layer, batch_size, seq_len
                    ),
                },
            )


def run_full_attention_benchmarks(
    rows: list[dict[str, object]],
    *,
    args: argparse.Namespace,
    config: Qwen3_5TextConfig,
    model_dtype: torch.dtype,
) -> None:
    batch_size = 1
    full_bench = FullAttentionBench(
        config,
        model_dtype,
        max_batch_size=batch_size,
        max_total_len=max(args.full_prefix_lens) + max(args.full_seq_lens),
    )
    for seq_len in args.full_seq_lens:
        for prefix_len in args.full_prefix_lens:
            stats = full_bench.bench(
                batch_size,
                seq_len,
                prefix_len,
                warmup_iters=args.warmup_iters,
                bench_iters=args.bench_iters,
            )
            total_q_tokens = batch_size * seq_len
            add_row(
                rows,
                category="full_attention_extend",
                stats=stats,
                payload={
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "prefix_len": prefix_len,
                    "q_tokens": total_q_tokens,
                    "tokens_per_s": total_q_tokens / (stats["median_ms"] / 1000.0),
                    "shape_info": full_attention_shape_info(
                        config, batch_size, seq_len, prefix_len
                    ),
                },
            )
