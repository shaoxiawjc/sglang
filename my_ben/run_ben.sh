
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
    --model-path /home/wjc/resources/models/qwen3_5_9b \
    --warmup-iters 20 \
    --bench-iters 100 \
    --batch-sizes 1 \
    --seq-lens 64 128 256 512 1024 \
    --prefix-lens 0 1024 2048 4096 8192 16384 \
    --linear-attn-backend triton

# python my_ben/bench_qwen35_overlap_recovery.py \
#     --model-path /home/wjc/resources/models/qwen3_5_9b \
#     --linear-layer-count 3 \
#     --full-layer-count 1 \
#     --linear-recompute-counts 0 1 2 3 \
#     --batch-sizes 1 4 \
#     --seq-lens 4096 2048 \
#     --prefix-lens 0 4096 8192 16384 32768 \
#     --warmup-iters 10 \
#     --bench-iters 20 \
#     --linear-attn-backend triton