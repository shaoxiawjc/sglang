export CUDA_VISIBLE_DEVICES=1

# python my_ben/bench_qwen35_hybrid_recovery.py \
#     --model-path /home/wjc/resources/models/qwen3_5_9b \
#     --warmup-iters 25 \
#     --bench-iters 100 \
#     --kv-token-counts 64 128 256 512 1024 2048 4096 8192 \
#     --state-slot-counts 1 2 4 8 16 \
#     --linear-batch-sizes 1 2 4 8 16 \
#     --linear-seq-lens 64 128 256 512 1024 2048 \
#     --full-seq-lens 32 64 128 256 512 1024 2048 \
#     --full-prefix-lens 0 1024 2048 4096 8192 16384 \
#     --linear-attn-backend triton

python my_ben/bench_qwen35_block_forward.py \
    --warmup-iters 20 \
    --bench-iters 100 \
    --batch-sizes 1 \
    --seq-lens 2048 4096 8192 16384 \
    --prefix-lens 0 512 1024 2048 4096 \
    --linear-attn-backend triton

python3 my_ben/bench_qwen35_overlap_recovery.py \
    --model-path /home/wjc/resources/models/qwen3_5_9b \
    --group-index 0 \
    --linear-layer-count 3 \
    --causal-layer-count 1 \
    --linear-recompute-counts 1 2 3 \
    --batch-sizes 1 \
    --seq-lens 2048 4096 8192 16384 \
    --prefix-lens 0 64 128 256 512 1024 2048 4096 \
    --warmup-iters 10 \
    --bench-iters 100 \
    --linear-attn-backend triton

# python3 my_ben/plot_qwen35_overlap_recovery.py \
#     --input-dir my_ben/results/qwen35_overlap_recovery/20260319-085232

# python3 my_ben/plot_qwen35_overlap_recovery.py \
#     --input-dir my_ben/results/qwen35_overlap_recovery/20260319-091717

python3 my_ben/profile_qwen35_overlap_recovery.py \
    --model-path /home/wjc/resources/models/qwen3_5_9b \
    --strategy ours_la_recompute_overlap_ca_kvcache \
    --linear-recompute-count 2 \
    --batch-size 1 \
    --seq-len 2048 \
    --prefix-len 1024 \
    --warmup-iters 20 \
    --profile-iters 1 \
    --linear-attn-backend triton \
    --trace-path my_ben/results/qwen35_overlap_recovery/trace_la_first_r2_pl1024.json

python3 my_ben/profile_qwen35_overlap_recovery.py \
    --model-path /home/wjc/resources/models/qwen3_5_9b \
    --strategy ours_ca_recompute_overlap_la_state_conv \
    --linear-recompute-count 2 \
    --batch-size 1 \
    --seq-len 2048 \
    --prefix-len 1024 \
    --warmup-iters 20 \
    --profile-iters 1 \
    --linear-attn-backend triton \
    --trace-path my_ben/results/qwen35_overlap_recovery/trace_ca_rcp_3la_pf_pl1024.json
