from typing import Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    get_linear_attn_decode_backend,
    get_linear_attn_prefill_backend,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.utils import get_bool_env_var, is_cpu, is_cuda, is_npu
from sglang.srt.utils.common import rank0_log

if not is_cpu():
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        CHUNK_SIZE as FLA_CHUNK_SIZE,
    )

if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn as causal_conv1d_fn_cuda,
    )

    causal_conv1d_fn = causal_conv1d_fn_cuda
elif is_npu():
    from sgl_kernel_npu.fla.fused_gdn_gating import fused_gdn_gating_npu
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
    )

    fused_gdn_gating = fused_gdn_gating_npu
    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu
elif is_cpu():
    from sgl_kernel.mamba import causal_conv1d_fn_cpu, causal_conv1d_update_cpu

    causal_conv1d_fn = causal_conv1d_fn_cpu
    causal_conv1d_update = causal_conv1d_update_cpu
    fused_gdn_gating = torch.ops.sgl_kernel.fused_gdn_gating_cpu


# RRMC fused boundary is the default path. Set SGLANG_RRMC_FUSED_BOUNDARY=0
# (or false) to fall back to the segmented boundary implementation.
_ENABLE_RRMC_FUSED_BOUNDARY = get_bool_env_var(
    "SGLANG_RRMC_FUSED_BOUNDARY", default="true"
)


class GDNKernelDispatcher:
    """Dispatches GDN kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
    ):
        triton_kernel = TritonGDNKernel()

        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("GDN CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                CuteDSLGDNKernel,
            )

            self.decode_kernel = CuteDSLGDNKernel()
        elif decode_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                FlashInferGDNKernel,
            )

            flashinfer_kernel = FlashInferGDNKernel()
            self.decode_kernel = flashinfer_kernel
        else:
            raise ValueError(f"Unsupported GDN decode backend: {decode_backend}")

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_cutedsl():
            raise ValueError(
                "CuTe DSL backend only supports decode, not prefill. "
                "Use --linear-attn-prefill-backend triton instead."
            )
        elif prefill_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            # Reuse the FlashInfer kernel if already created for decode
            if decode_backend.is_flashinfer():
                self.extend_kernel = flashinfer_kernel
            else:
                from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                    FlashInferGDNKernel,
                )

                flashinfer_kernel = FlashInferGDNKernel()
                self.extend_kernel = flashinfer_kernel
        else:
            raise ValueError(f"Unsupported GDN prefill backend: {prefill_backend}")

        # Verify kernel: use FlashInfer if either decode or prefill selected it
        if decode_backend.is_flashinfer() or prefill_backend.is_flashinfer():
            self.verify_kernel = flashinfer_kernel
        else:
            self.verify_kernel = triton_kernel

        self.supports_packed_decode = getattr(
            self.decode_kernel, "supports_packed_decode", False
        )

        rank0_log(
            f"GDN kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__}, "
            f"verify={self.verify_kernel.__class__.__name__} "
            f"packed_decode={self.supports_packed_decode}"
        )

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Attempt packed decode. Returns output tensor or None if
        the decode kernel does not support packed decode."""
        if not self.supports_packed_decode:
            return None
        return self.decode_kernel.packed_decode(
            mixed_qkv,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            **kwargs,
        )

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.decode_kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        return self.extend_kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.verify_kernel.target_verify(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )


class GDNAttnBackend(MambaAttnBackendBase):
    """Attention backend for GDN (Gated Delta Network) linear attention."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )
        if not is_cpu() and not is_npu():
            assert (
                self.conv_states_shape[-1] < FLA_CHUNK_SIZE
            ), f"{self.conv_states_shape[-1]=} should be less than {FLA_CHUNK_SIZE}"

        decode_backend = get_linear_attn_decode_backend()
        prefill_backend = get_linear_attn_prefill_backend()
        self.kernel_dispatcher = GDNKernelDispatcher(decode_backend, prefill_backend)

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        assert isinstance(mixed_qkv, torch.Tensor)
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices,
        )

        # Skip split + reshape + separate gating kernel by consuming
        # the packed mixed_qkv directly in a single fused Triton kernel.
        if self.kernel_dispatcher.supports_packed_decode:
            core_attn_out = self.kernel_dispatcher.packed_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                scale=layer.head_k_dim**-0.5,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                num_v_heads=layer.num_v_heads,
                head_v_dim=layer.head_v_dim,
            )
            self._track_mamba_state_decode(
                forward_batch, conv_states, ssm_states, cache_indices
            )
            return core_attn_out

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        # Reshape from [bs, h*d] to [1, bs, h, d]
        bs = forward_batch.batch_size
        query = query.view(1, bs, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, bs, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, bs, layer.num_v_heads, layer.head_v_dim)

        core_attn_out = self.kernel_dispatcher.decode(
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices
        )

        return core_attn_out

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]

        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata

        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices
        retrieve_next_token = forward_metadata.retrieve_next_token
        retrieve_next_sibling = forward_metadata.retrieve_next_sibling
        retrieve_parent_token = forward_metadata.retrieve_parent_token

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            has_initial_states = torch.ones(
                seq_len // forward_batch.spec_info.draft_token_num,
                dtype=torch.bool,
                device=forward_batch.input_ids.device,
            )
            intermediate_state_indices = torch.arange(
                cache_indices.shape[0], dtype=torch.int32, device=cache_indices.device
            )
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0

        if (
            not is_target_verify
            and getattr(forward_batch, "rrmc_enabled", False)
            and forward_batch.rrmc_boundary_local_ends_cpu
        ):
            return self._forward_extend_rrmc_boundaries(
                layer=layer,
                forward_batch=forward_batch,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                conv_states=conv_states,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
            )

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)
            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_states,
                layer.conv_weights,
                layer.bias,
                layer.activation,
                conv_state_indices=cache_indices[:batch_size],
                intermediate_conv_window=intermediate_conv_window_cache,
                intermediate_state_indices=intermediate_state_indices[:batch_size],
                retrieve_next_token=retrieve_next_token,
                retrieve_next_sibling=retrieve_next_sibling,
                retrieve_parent_token=retrieve_parent_token,
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            mixed_qkv = mixed_qkv.transpose(0, 1)
            if (
                forward_batch.mamba_track_mask is not None
                and forward_batch.mamba_track_mask.any()
            ):
                conv_dst = forward_batch.mamba_track_indices
                mixed_qkv_to_track = mixed_qkv[
                    :, forward_metadata.track_conv_indices
                ].transpose(0, 1)
                mask_indices = forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
                conv_states[conv_dst[mask_indices]] = mixed_qkv_to_track

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                layer.conv_weights,
                layer.bias,
                activation=layer.activation,
                conv_states=conv_states,
                has_initial_state=has_initial_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ).transpose(0, 1)[:seq_len]

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )

        actual_seq_len = query.shape[0]
        query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)

        if is_target_verify:
            core_attn_out = self.kernel_dispatcher.target_verify(
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                q=query,
                k=key,
                v=value,
                a=a,
                b=b,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                intermediate_states_buffer=intermediate_state_cache,
                intermediate_state_indices=intermediate_state_indices,
                cache_steps=forward_batch.spec_info.draft_token_num,
                retrieve_parent_token=retrieve_parent_token,
            )
        else:
            g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
            core_attn_out, last_recurrent_state, h = self.kernel_dispatcher.extend(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
            )

            if (is_npu() or is_cpu()) and last_recurrent_state is not None:
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state

            if h is not None:
                self._track_mamba_state_extend(
                    forward_batch, h, ssm_states, forward_metadata
                )

        return core_attn_out

    def _forward_extend_rrmc_boundaries(
        self,
        *,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = mixed_qkv.shape[0]
        g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)

        boundaries_by_req: dict[int, list[tuple[int, int, int]]] = {}
        for req_idx, local_start, local_end, mamba_idx in zip(
            forward_batch.rrmc_boundary_req_indices_cpu,
            forward_batch.rrmc_boundary_local_starts_cpu,
            forward_batch.rrmc_boundary_local_ends_cpu,
            forward_batch.rrmc_boundary_mamba_indices_cpu,
        ):
            boundaries_by_req.setdefault(int(req_idx), []).append(
                (int(local_start), int(local_end), int(mamba_idx))
            )
        for req_boundaries in boundaries_by_req.values():
            req_boundaries.sort(key=lambda item: (item[0], item[1]))

        fused_out = self._try_forward_extend_rrmc_fused_boundaries(
            layer=layer,
            forward_batch=forward_batch,
            mixed_qkv=mixed_qkv,
            g=g,
            beta=beta,
            conv_states=conv_states,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            boundaries_by_req=boundaries_by_req,
        )
        if fused_out is not None:
            return fused_out

        core_attn_out = mixed_qkv.new_empty(
            1, seq_len, layer.num_v_heads, layer.head_v_dim
        )

        segments_by_req: list[list[tuple[int, int, int, bool, int]]] = []
        req_start = 0
        for req_idx, req_extend_len in enumerate(forward_batch.extend_seq_lens_cpu):
            req_end = req_start + int(req_extend_len)
            segment_start = req_start
            req_boundaries = boundaries_by_req.get(req_idx, [])
            has_initial_state = (
                int(forward_batch.extend_prefix_lens_cpu[req_idx]) > 0
            )
            req_segments: list[tuple[int, int, int, bool, int]] = []

            for boundary_start, boundary_end, boundary_mamba_idx in req_boundaries:
                if boundary_start > segment_start:
                    req_segments.append(
                        (
                            req_idx,
                            segment_start,
                            boundary_start,
                            has_initial_state,
                            -1,
                        )
                    )
                    has_initial_state = True
                    segment_start = boundary_start

                if boundary_end > segment_start:
                    req_segments.append(
                        (
                            req_idx,
                            segment_start,
                            boundary_end,
                            has_initial_state,
                            boundary_mamba_idx,
                        )
                    )
                    has_initial_state = True
                    segment_start = boundary_end

            if segment_start < req_end:
                req_segments.append(
                    (
                        req_idx,
                        segment_start,
                        req_end,
                        has_initial_state,
                        -1,
                    )
                )

            segments_by_req.append(req_segments)
            req_start = req_end

        assert req_start == seq_len

        max_num_segments = max(
            (len(req_segments) for req_segments in segments_by_req), default=0
        )
        for segment_idx in range(max_num_segments):
            segment_batch = [
                req_segments[segment_idx]
                for req_segments in segments_by_req
                if segment_idx < len(req_segments)
            ]
            self._run_rrmc_prefill_segment_batch(
                layer=layer,
                mixed_qkv=mixed_qkv,
                g=g,
                beta=beta,
                conv_states=conv_states,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                segments=segment_batch,
                core_attn_out=core_attn_out,
            )

            capture_req_indices = []
            capture_mamba_indices = []
            for req_idx, _, _, _, boundary_mamba_idx in segment_batch:
                if boundary_mamba_idx >= 0:
                    capture_req_indices.append(req_idx)
                    capture_mamba_indices.append(boundary_mamba_idx)
            self._capture_rrmc_boundary_states(
                conv_states=conv_states,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                req_indices=capture_req_indices,
                boundary_mamba_indices=capture_mamba_indices,
            )

        return core_attn_out

    def _try_forward_extend_rrmc_fused_boundaries(
        self,
        *,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        boundaries_by_req: dict[int, list[tuple[int, int, int]]],
    ) -> Optional[torch.Tensor]:
        if not _ENABLE_RRMC_FUSED_BOUNDARY:
            return None
        if not is_cuda() or not isinstance(
            self.kernel_dispatcher.extend_kernel, TritonGDNKernel
        ):
            return None

        metadata_cache = getattr(
            forward_batch, "_rrmc_fused_boundary_metadata_cache", None
        )
        if metadata_cache is None:
            metadata_cache = {}
            setattr(
                forward_batch,
                "_rrmc_fused_boundary_metadata_cache",
                metadata_cache,
            )
        metadata_key = (str(mixed_qkv.device), int(conv_states.shape[-1]))
        if metadata_key not in metadata_cache:
            metadata_cache[metadata_key] = self._build_rrmc_fused_boundary_metadata(
                forward_batch=forward_batch,
                mixed_qkv=mixed_qkv,
                conv_states=conv_states,
                boundaries_by_req=boundaries_by_req,
            )
        fused_metadata = metadata_cache[metadata_key]
        if fused_metadata is None:
            return None

        (
            boundary_state_indices_by_chunk,
            boundary_token_offsets_by_chunk,
            boundary_conv_indices,
            boundary_mamba_indices,
        ) = fused_metadata

        raw_mixed_qkv = mixed_qkv
        mixed_qkv = causal_conv1d_fn(
            raw_mixed_qkv.transpose(0, 1),
            layer.conv_weights,
            layer.bias,
            activation=layer.activation,
            conv_states=conv_states,
            has_initial_state=forward_batch.extend_prefix_lens > 0,
            cache_indices=cache_indices,
            query_start_loc=self.forward_metadata.query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)[: raw_mixed_qkv.shape[0]]

        self._capture_rrmc_boundary_conv_windows(
            mixed_qkv=raw_mixed_qkv,
            conv_states=conv_states,
            boundary_conv_indices=boundary_conv_indices,
            boundary_mamba_indices=boundary_mamba_indices,
        )

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        seq_len = query.shape[0]
        query = query.view(1, seq_len, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, seq_len, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, seq_len, layer.num_v_heads, layer.head_v_dim)

        core_attn_out, last_recurrent_state, _ = self.kernel_dispatcher.extend(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=self.forward_metadata.query_start_loc,
            rrmc_boundary_state_indices_by_chunk=boundary_state_indices_by_chunk,
            rrmc_boundary_token_offsets_by_chunk=boundary_token_offsets_by_chunk,
        )

        if last_recurrent_state is not None:
            last_recurrent_state = last_recurrent_state.to(ssm_states.dtype, copy=False)
            ssm_states[cache_indices] = last_recurrent_state
        return core_attn_out

    def _build_rrmc_fused_boundary_metadata(
        self,
        *,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        conv_states: torch.Tensor,
        boundaries_by_req: dict[int, list[tuple[int, int, int]]],
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        conv_state_len = int(conv_states.shape[-1])
        req_starts = []
        req_chunk_offsets = []
        req_start = 0
        chunk_offset = 0
        for req_extend_len in forward_batch.extend_seq_lens_cpu:
            req_extend_len = int(req_extend_len)
            req_starts.append(req_start)
            req_chunk_offsets.append(chunk_offset)
            chunk_offset += (req_extend_len + FLA_CHUNK_SIZE - 1) // FLA_CHUNK_SIZE
            req_start += req_extend_len

        if chunk_offset <= 0:
            return None

        state_indices_by_chunk = [-1] * chunk_offset
        token_offsets_by_chunk = [0] * chunk_offset
        boundary_conv_indices = []
        boundary_mamba_indices = []

        for req_idx, req_boundaries in boundaries_by_req.items():
            req_start = req_starts[req_idx]
            req_extend_len = int(forward_batch.extend_seq_lens_cpu[req_idx])
            for _, boundary_end, boundary_mamba_idx in req_boundaries:
                boundary_extend_end = int(boundary_end) - req_start
                if boundary_extend_end <= 0 or boundary_extend_end > req_extend_len:
                    return None
                if boundary_extend_end < conv_state_len:
                    return None

                chunk_idx = (boundary_extend_end - 1) // FLA_CHUNK_SIZE
                global_chunk_idx = req_chunk_offsets[req_idx] + chunk_idx
                if state_indices_by_chunk[global_chunk_idx] >= 0:
                    return None

                state_indices_by_chunk[global_chunk_idx] = int(boundary_mamba_idx)
                token_offsets_by_chunk[global_chunk_idx] = (
                    boundary_extend_end - chunk_idx * FLA_CHUNK_SIZE
                )
                boundary_conv_indices.append(
                    list(range(int(boundary_end) - conv_state_len, int(boundary_end)))
                )
                boundary_mamba_indices.append(int(boundary_mamba_idx))

        if not boundary_mamba_indices:
            return None

        return (
            torch.tensor(
                state_indices_by_chunk,
                dtype=torch.long,
                device=mixed_qkv.device,
            ),
            torch.tensor(
                token_offsets_by_chunk,
                dtype=torch.int32,
                device=mixed_qkv.device,
            ),
            torch.tensor(
                boundary_conv_indices,
                dtype=torch.long,
                device=mixed_qkv.device,
            ),
            torch.tensor(
                boundary_mamba_indices,
                dtype=torch.long,
                device=mixed_qkv.device,
            ),
        )

    def _capture_rrmc_boundary_conv_windows(
        self,
        *,
        mixed_qkv: torch.Tensor,
        conv_states: torch.Tensor,
        boundary_conv_indices: torch.Tensor,
        boundary_mamba_indices: torch.Tensor,
    ) -> None:
        if boundary_mamba_indices.numel() == 0:
            return

        raw_mixed_qkv = mixed_qkv.transpose(0, 1)
        conv_states[boundary_mamba_indices] = raw_mixed_qkv[
            :, boundary_conv_indices
        ].transpose(0, 1)

    def _run_rrmc_prefill_segment_batch(
        self,
        *,
        layer: RadixLinearAttention,
        mixed_qkv: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        segments: list[tuple[int, int, int, bool, int]],
        core_attn_out: torch.Tensor,
    ) -> None:
        segment_lens = [
            int(segment_end - segment_start)
            for _, segment_start, segment_end, _, _ in segments
        ]
        total_segment_len = sum(segment_lens)
        if total_segment_len <= 0:
            return

        segment_query_start_locs = [0]
        running_len = 0
        for segment_len in segment_lens:
            running_len += segment_len
            segment_query_start_locs.append(running_len)

        req_indices = torch.tensor(
            [req_idx for req_idx, _, _, _, _ in segments],
            dtype=torch.long,
            device=cache_indices.device,
        )
        segment_cache_indices = cache_indices[req_indices]
        segment_query_start_loc = torch.tensor(
            segment_query_start_locs,
            dtype=torch.int32,
            device=mixed_qkv.device,
        )
        segment_has_initial_state = torch.tensor(
            [has_initial_state for _, _, _, has_initial_state, _ in segments],
            dtype=torch.bool,
            device=mixed_qkv.device,
        )

        if len(segments) == 1:
            _, segment_start, segment_end, _, _ = segments[0]
            packed_mixed_qkv = mixed_qkv[segment_start:segment_end]
            packed_g = g[:, segment_start:segment_end]
            packed_beta = beta[:, segment_start:segment_end]
        else:
            packed_mixed_qkv = torch.cat(
                [
                    mixed_qkv[segment_start:segment_end]
                    for _, segment_start, segment_end, _, _ in segments
                ],
                dim=0,
            )
            packed_g = torch.cat(
                [
                    g[:, segment_start:segment_end]
                    for _, segment_start, segment_end, _, _ in segments
                ],
                dim=1,
            )
            packed_beta = torch.cat(
                [
                    beta[:, segment_start:segment_end]
                    for _, segment_start, segment_end, _, _ in segments
                ],
                dim=1,
            )

        packed_mixed_qkv = causal_conv1d_fn(
            packed_mixed_qkv.transpose(0, 1),
            layer.conv_weights,
            layer.bias,
            activation=layer.activation,
            conv_states=conv_states,
            has_initial_state=segment_has_initial_state,
            cache_indices=segment_cache_indices,
            query_start_loc=segment_query_start_loc,
            seq_lens_cpu=segment_lens,
        ).transpose(0, 1)[:total_segment_len]

        query, key, value = torch.split(
            packed_mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        query = query.view(1, total_segment_len, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, total_segment_len, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, total_segment_len, layer.num_v_heads, layer.head_v_dim)

        segment_out, last_recurrent_state, _ = self.kernel_dispatcher.extend(
            q=query,
            k=key,
            v=value,
            g=packed_g,
            beta=packed_beta,
            ssm_states=ssm_states,
            cache_indices=segment_cache_indices,
            query_start_loc=segment_query_start_loc,
        )

        if (is_npu() or is_cpu()) and last_recurrent_state is not None:
            last_recurrent_state = last_recurrent_state.to(
                ssm_states.dtype, copy=False
            )
            ssm_states[segment_cache_indices] = last_recurrent_state

        segment_offset = 0
        for (_, segment_start, segment_end, _, _), segment_len in zip(
            segments, segment_lens
        ):
            next_segment_offset = segment_offset + segment_len
            core_attn_out[:, segment_start:segment_end] = segment_out[
                :, segment_offset:next_segment_offset
            ]
            segment_offset = next_segment_offset

    def _capture_rrmc_boundary_states(
        self,
        *,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        req_indices: list[int],
        boundary_mamba_indices: list[int],
    ) -> None:
        if not boundary_mamba_indices:
            return

        req_indices_tensor = torch.tensor(
            req_indices,
            dtype=torch.long,
            device=cache_indices.device,
        )
        src_index = cache_indices[req_indices_tensor].to(dtype=torch.long)
        dst_index = torch.tensor(
            boundary_mamba_indices,
            dtype=torch.long,
            device=cache_indices.device,
        )
        conv_states.index_copy_(0, dst_index, conv_states.index_select(0, src_index))
        ssm_states.index_copy_(0, dst_index, ssm_states.index_select(0, src_index))
