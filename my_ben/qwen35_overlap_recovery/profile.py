from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "FLASHINFER_WORKSPACE_BASE", str(REPO_ROOT / ".flashinfer_workspace")
)
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT))

from my_ben.qwen35_hybrid_recovery.config import (
    DTYPE_MAP,
    DEFAULT_QWEN35_FULL_ATTENTION_INTERVAL,
    DEFAULT_QWEN35_HEAD_DIM,
    DEFAULT_QWEN35_LAYER_TYPES,
    DEFAULT_QWEN35_LINEAR_CONV_KERNEL_DIM,
    DEFAULT_QWEN35_LINEAR_KEY_HEAD_DIM,
    DEFAULT_QWEN35_LINEAR_NUM_KEY_HEADS,
    DEFAULT_QWEN35_LINEAR_NUM_VALUE_HEADS,
    DEFAULT_QWEN35_LINEAR_VALUE_HEAD_DIM,
    DEFAULT_QWEN35_MAMBA_SSM_DTYPE,
    DEFAULT_QWEN35_MODEL_DTYPE,
    DEFAULT_QWEN35_NUM_ATTENTION_HEADS,
    DEFAULT_QWEN35_NUM_HIDDEN_LAYERS,
    DEFAULT_QWEN35_NUM_KEY_VALUE_HEADS,
)
from sglang.srt.layers.attention.linear.utils import initialize_linear_attn_config
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

from .benches import CausalBlockBench, LinearBlockGroupBench, _reset_group
from .config import resolve_model_and_group


@dataclass
class ProfileRuntime:
    linear_bench: LinearBlockGroupBench
    causal_bench: CausalBlockBench
    recompute_stream: torch.cuda.Stream
    prefetch_stream: torch.cuda.Stream
    linear_positions: list[int]


@dataclass
class StrategyContexts:
    batch_size: int
    seq_len: int
    prefix_len: int
    total_len: int
    linear_total_context: object
    linear_current_context: object
    causal_total_context: object
    causal_current_context: object


@contextmanager
def trace_range(name: str):
    with record_function(name):
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()


def parse_args(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/home/wjc/resources/models/qwen3_5_9b"),
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=repo_root
        / "my_ben"
        / "results"
        / "qwen35_overlap_recovery"
        / "profile_trace.json",
        help="Output Chrome trace JSON path. Load this file in Perfetto.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=[
            "baseline_recompute",
            "baseline_offload_onload",
            "ours_ca_recompute_overlap_la_state_conv",
            "ours_la_recompute_overlap_ca_kvcache",
        ],
        default="ours_ca_recompute_overlap_la_state_conv",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--prefix-len", type=int, default=8192)
    parser.add_argument(
        "--linear-recompute-count",
        type=int,
        default=1,
        help="Used only for ours_la_recompute_overlap_ca_kvcache.",
    )
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--profile-iters", type=int, default=1)
    parser.add_argument("--group-index", type=int, default=0)
    parser.add_argument("--linear-layer-count", type=int, default=3)
    parser.add_argument("--causal-layer-count", type=int, default=1)
    parser.add_argument(
        "--linear-recompute-counts",
        type=int,
        nargs="+",
        default=[1, 2, 3],
    )
    parser.add_argument(
        "--num-hidden-layers", type=int, default=DEFAULT_QWEN35_NUM_HIDDEN_LAYERS
    )
    parser.add_argument(
        "--num-attention-heads", type=int, default=DEFAULT_QWEN35_NUM_ATTENTION_HEADS
    )
    parser.add_argument(
        "--num-key-value-heads", type=int, default=DEFAULT_QWEN35_NUM_KEY_VALUE_HEADS
    )
    parser.add_argument("--head-dim", type=int, default=DEFAULT_QWEN35_HEAD_DIM)
    parser.add_argument(
        "--full-attention-interval",
        type=int,
        default=DEFAULT_QWEN35_FULL_ATTENTION_INTERVAL,
    )
    parser.add_argument(
        "--layer-types",
        type=str,
        nargs="+",
        default=DEFAULT_QWEN35_LAYER_TYPES,
    )
    parser.add_argument(
        "--linear-conv-kernel-dim",
        type=int,
        default=DEFAULT_QWEN35_LINEAR_CONV_KERNEL_DIM,
    )
    parser.add_argument(
        "--linear-num-key-heads",
        type=int,
        default=DEFAULT_QWEN35_LINEAR_NUM_KEY_HEADS,
    )
    parser.add_argument(
        "--linear-num-value-heads",
        type=int,
        default=DEFAULT_QWEN35_LINEAR_NUM_VALUE_HEADS,
    )
    parser.add_argument(
        "--linear-key-head-dim",
        type=int,
        default=DEFAULT_QWEN35_LINEAR_KEY_HEAD_DIM,
    )
    parser.add_argument(
        "--linear-value-head-dim",
        type=int,
        default=DEFAULT_QWEN35_LINEAR_VALUE_HEAD_DIM,
    )
    parser.add_argument(
        "--model-dtype",
        choices=sorted(DTYPE_MAP.keys()),
        default=DEFAULT_QWEN35_MODEL_DTYPE,
    )
    parser.add_argument(
        "--mamba-conv-dtype",
        choices=sorted(DTYPE_MAP.keys()),
        default=None,
    )
    parser.add_argument(
        "--mamba-ssm-dtype",
        choices=sorted(DTYPE_MAP.keys()),
        default=DEFAULT_QWEN35_MAMBA_SSM_DTYPE,
    )
    parser.add_argument(
        "--linear-attn-backend",
        type=str,
        choices=["triton", "cutedsl", "flashinfer"],
        default="flashinfer",
    )
    parser.add_argument(
        "--linear-attn-decode-backend",
        type=str,
        choices=["triton", "cutedsl", "flashinfer"],
        default=None,
    )
    parser.add_argument(
        "--linear-attn-prefill-backend",
        type=str,
        choices=["triton", "cutedsl", "flashinfer"],
        default=None,
    )
    return parser.parse_args()


def build_runtime(
    args: argparse.Namespace,
    *,
    config,
    model_dtype: torch.dtype,
    linear_cache_params,
    linear_layer_ids: list[int],
    causal_layer_id: int,
) -> ProfileRuntime:
    linear_bench = LinearBlockGroupBench(
        config,
        model_dtype,
        linear_cache_params,
        linear_layer_ids,
        max_batch_size=args.batch_size,
        max_context_len=args.prefix_len + args.seq_len,
    )
    causal_bench = CausalBlockBench(
        config,
        model_dtype,
        causal_layer_id,
        max_batch_size=args.batch_size,
        max_total_len=args.prefix_len + args.seq_len,
    )
    return ProfileRuntime(
        linear_bench=linear_bench,
        causal_bench=causal_bench,
        recompute_stream=torch.cuda.Stream(),
        prefetch_stream=torch.cuda.Stream(),
        linear_positions=list(range(len(linear_layer_ids))),
    )


def build_contexts(runtime: ProfileRuntime, args: argparse.Namespace) -> StrategyContexts:
    total_len = args.prefix_len + args.seq_len
    return StrategyContexts(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        prefix_len=args.prefix_len,
        total_len=total_len,
        linear_total_context=runtime.linear_bench.make_context(args.batch_size, total_len, 0),
        linear_current_context=runtime.linear_bench.make_context(
            args.batch_size, args.seq_len, args.prefix_len
        ),
        causal_total_context=runtime.causal_bench.make_context(args.batch_size, total_len, 0),
        causal_current_context=runtime.causal_bench.make_context(
            args.batch_size, args.seq_len, args.prefix_len
        ),
    )


def _run_baseline_recompute_profiled(
    runtime: ProfileRuntime,
    contexts: StrategyContexts,
) -> None:
    with trace_range("baseline.recompute_all_la"):
        runtime.linear_bench.run_context(
            runtime.linear_positions, contexts.linear_total_context
        )
    with trace_range("baseline.recompute_ca"):
        runtime.causal_bench.run_context(contexts.causal_total_context)


def _run_baseline_offload_onload_profiled(
    runtime: ProfileRuntime,
    contexts: StrategyContexts,
) -> None:
    with trace_range("baseline.onload_all_la_state"):
        runtime.linear_bench.prefetch_conv(runtime.linear_positions, contexts.batch_size)
    with trace_range("baseline.onload_ca_kvcache"):
        runtime.causal_bench.prefetch_prefix_kv(contexts.causal_current_context)
    with trace_range("baseline.forward_all_la_current"):
        runtime.linear_bench.run_context(
            runtime.linear_positions, contexts.linear_current_context
        )
    with trace_range("baseline.forward_ca_current"):
        runtime.causal_bench.run_context(contexts.causal_current_context)


def _run_ours_ca_first_profiled(
    runtime: ProfileRuntime,
    contexts: StrategyContexts,
) -> None:
    recompute_done = torch.cuda.Event()
    prefetch_events: list[torch.cuda.Event] = []

    with torch.cuda.stream(runtime.recompute_stream):
        with trace_range("recompute.ca_total"):
            runtime.causal_bench.run_context(contexts.causal_total_context)
        recompute_done.record()

    with torch.cuda.stream(runtime.prefetch_stream):
        for position in runtime.linear_positions:
            with trace_range(f"prefetch.la_state.layer_{position}"):
                runtime.linear_bench.prefetch_conv([position], contexts.batch_size)
            prefetch_done = torch.cuda.Event()
            prefetch_done.record()
            prefetch_events.append(prefetch_done)

    current_stream = torch.cuda.current_stream()
    with trace_range("wait.ca_recompute_done"):
        current_stream.wait_event(recompute_done)

    with trace_range("prepare.la_current_context"):
        runtime.linear_bench.prepare_context(contexts.linear_current_context)
    hidden_states = contexts.linear_current_context.hidden_states
    residual = None
    for position, prefetch_done in zip(runtime.linear_positions, prefetch_events):
        with trace_range(f"wait.la_prefetch_done.layer_{position}"):
            current_stream.wait_event(prefetch_done)
        with trace_range(f"forward.la_current.layer_{position}"):
            hidden_states, residual = runtime.linear_bench.run_one_layer(
                position,
                hidden_states,
                residual,
                contexts.linear_current_context,
            )


def _run_ours_la_first_profiled(
    runtime: ProfileRuntime,
    contexts: StrategyContexts,
    linear_recompute_count: int,
) -> None:
    recompute_positions = runtime.linear_positions[:linear_recompute_count]
    onload_positions = runtime.linear_positions[linear_recompute_count:]
    kvcache_done = torch.cuda.Event()
    recompute_done = torch.cuda.Event()

    with torch.cuda.stream(runtime.recompute_stream):
        with trace_range(f"recompute.la_total.count_{linear_recompute_count}"):
            runtime.linear_bench.run_context(
                recompute_positions, contexts.linear_total_context
            )
        recompute_done.record()

    with torch.cuda.stream(runtime.prefetch_stream):
        with trace_range("prefetch.ca_kvcache"):
            runtime.causal_bench.prefetch_prefix_kv(contexts.causal_current_context)
        kvcache_done.record()

    current_stream = torch.cuda.current_stream()
    with trace_range("wait.ca_kvcache_done"):
        current_stream.wait_event(kvcache_done)
    with trace_range("wait.la_recompute_done"):
        current_stream.wait_event(recompute_done)

    if onload_positions:
        prefetch_events: list[torch.cuda.Event] = []
        with torch.cuda.stream(runtime.prefetch_stream):
            for position in onload_positions:
                with trace_range(f"prefetch.la_state.layer_{position}"):
                    runtime.linear_bench.prefetch_conv([position], contexts.batch_size)
                prefetch_done = torch.cuda.Event()
                prefetch_done.record()
                prefetch_events.append(prefetch_done)

        with trace_range("prepare.la_current_context"):
            runtime.linear_bench.prepare_context(contexts.linear_current_context)
        hidden_states = contexts.linear_current_context.hidden_states
        residual = None
        for position, prefetch_done in zip(onload_positions, prefetch_events):
            with trace_range(f"wait.la_prefetch_done.layer_{position}"):
                current_stream.wait_event(prefetch_done)
            with trace_range(f"forward.la_current.layer_{position}"):
                hidden_states, residual = runtime.linear_bench.run_one_layer(
                    position,
                    hidden_states,
                    residual,
                    contexts.linear_current_context,
                )

    with trace_range("forward.ca_current"):
        runtime.causal_bench.run_context(contexts.causal_current_context)


def run_profile_iteration(
    runtime: ProfileRuntime,
    contexts: StrategyContexts,
    args: argparse.Namespace,
) -> None:
    with trace_range("reset.caches"):
        _reset_group(runtime.linear_bench, runtime.causal_bench, contexts.batch_size)

    if args.strategy == "baseline_recompute":
        _run_baseline_recompute_profiled(runtime, contexts)
    elif args.strategy == "baseline_offload_onload":
        _run_baseline_offload_onload_profiled(runtime, contexts)
    elif args.strategy == "ours_ca_recompute_overlap_la_state_conv":
        _run_ours_ca_first_profiled(runtime, contexts)
    elif args.strategy == "ours_la_recompute_overlap_ca_kvcache":
        _run_ours_la_first_profiled(runtime, contexts, args.linear_recompute_count)
    else:
        raise ValueError(f"Unsupported strategy: {args.strategy}")

    with trace_range("iteration.cuda_synchronize"):
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args(REPO_ROOT)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for profiling.")

    server_args = ServerArgs(model_path="dummy")
    server_args.linear_attn_backend = args.linear_attn_backend
    server_args.linear_attn_decode_backend = args.linear_attn_decode_backend
    server_args.linear_attn_prefill_backend = args.linear_attn_prefill_backend
    set_global_server_args_for_scheduler(server_args)
    initialize_linear_attn_config(server_args)

    (
        config,
        model_dtype,
        _raw_config,
        linear_layer_ids,
        causal_layer_id,
        _recompute_counts,
        linear_cache_params,
    ) = resolve_model_and_group(args)
    config.torch_dtype = model_dtype

    runtime = build_runtime(
        args,
        config=config,
        model_dtype=model_dtype,
        linear_cache_params=linear_cache_params,
        linear_layer_ids=linear_layer_ids,
        causal_layer_id=causal_layer_id,
    )
    contexts = build_contexts(runtime, args)

    for _ in range(args.warmup_iters):
        run_profile_iteration(runtime, contexts, args)

    trace_path = args.trace_path.resolve()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(args.profile_iters):
            run_profile_iteration(runtime, contexts, args)

    prof.export_chrome_trace(str(trace_path))
    print(f"Saved Perfetto-compatible trace to {trace_path}")
    print("Open https://ui.perfetto.dev and load the exported JSON trace.")
