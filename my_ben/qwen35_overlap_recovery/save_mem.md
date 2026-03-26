# CPU Memory Saving for Overlap Recovery

本文描述当前 `my_ben/qwen35_overlap_recovery` 实现里，不同策略相对于单纯 `Offload-Onload` 基线节约了多少 CPU 内存。

这里按当前实现来统计 CPU 侧保存的恢复数据：

- Linear Attention 的 `conv state`
- Linear Attention 的 `temporal state`
- Causal Attention 的 `prefix KV cache`

不统计：

- GPU staging buffer
- hidden states / residual / 其他中间张量

## 1. 符号定义

设：

- $B$：batch size
- $P$：prefix length
- $N_{\mathrm{LA}} = 3$：Linear Attention 层数
- $N_{\mathrm{CA}} = 1$：Causal Attention 层数
- $a \in \{1, 2, 3\}$：在 `LA -> CA` 策略里，选择重计算的 LA 层数

再设：

- $S_{\mathrm{conv}}$：单个 LA layer、单个 batch slot 的 `conv state` 大小
- $S_{\mathrm{temp}}$：单个 LA layer、单个 batch slot 的 `temporal state` 大小
- $S_{\mathrm{state}} = S_{\mathrm{conv}} + S_{\mathrm{temp}}$
- $S_{\mathrm{kv}}$：单个 CA prefix token 的 KV cache 大小

## 2. 当前实现里的实际 cache 大小

### 2.1 LA 的 conv state

默认配置：

- `linear_conv_kernel_dim = 4`
- `linear_num_key_heads = 16`
- `linear_num_value_heads = 32`
- `linear_value_head_dim = 128`
- `mamba conv dtype = bfloat16`
- `tp_world_size = 1`
- `intermediate_size = linear_value_head_dim * linear_num_value_heads = 128 * 32 = 4096`
- `conv_kernel = linear_conv_kernel_dim = 4`

代进去：

$$
convdim = 4096 + 2 \times 16 \times 128 = 4096 + 4096 = 8192
$$

当前 `conv` shape 为：

$$
(8192, 3)
$$

因此：

$$
S_{\mathrm{conv}}
= 8192 \times 3 \times 2
= 49152 \text{ bytes}
= 48 \text{ KiB}
$$

### 2.2 LA 的 temporal state

默认配置下，`temporal` shape 为：

$$
(32, 128, 128)
$$

并且 `temporal dtype = float32`，因此：

$$
S_{\mathrm{temp}}
= 32 \times 128 \times 128 \times 4
= 2097152 \text{ bytes}
= 2 \text{ MiB}
$$

所以单个 LA layer、单个 batch slot 的总 state 大小为：

$$
S_{\mathrm{state}}
= S_{\mathrm{conv}} + S_{\mathrm{temp}}
= 49152 + 2097152
= 2146304 \text{ bytes}
$$

即：

$$
S_{\mathrm{state}} = 2.046875 \text{ MiB}
$$

### 2.3 CA 的 prefix KV cache

默认配置：

- `num_key_value_heads = 4`
- `head_dim = 256`
- KV dtype = `bfloat16`

每个 prefix token 需要一份 $K$ 和一份 $V$，因此：

$$
S_{\mathrm{kv}}
= 2 \times 4 \times 256 \times 2
= 4096 \text{ bytes}
= 4 \text{ KiB/token}
$$

## 3. Offload-Onload 基线

单纯 `Offload-Onload` 需要在 CPU 上保存：

- 3 个 LA layer 的完整 state
- 1 个 CA layer 的全部 prefix KV

因此基线 CPU 内存为：

$$
M_{\mathrm{offload}}
= B \cdot N_{\mathrm{LA}} \cdot S_{\mathrm{state}}
+ B \cdot P \cdot S_{\mathrm{kv}}
$$

代入 $N_{\mathrm{LA}} = 3$：

$$
M_{\mathrm{offload}}
= B \cdot (3S_{\mathrm{state}} + PS_{\mathrm{kv}})
$$

再代入数值：

$$
M_{\mathrm{offload}}
= B \cdot (3 \times 2146304 + 4096P)
$$

即：

$$
M_{\mathrm{offload}}
= B \cdot (6438912 + 4096P) \text{ bytes}
$$

由于：

$$
\frac{6438912}{4096} = 1572,\qquad \frac{2146304}{4096} = 524
$$

所以也可以写成：

$$
M_{\mathrm{offload}} = 4096B(P + 1572)
$$

## 4. 不同策略的 CPU 内存

### 4.1 Pure Recompute

完全不 offload，则：

$$
M_{\mathrm{recompute}} = 0
$$

相对于 `Offload-Onload` 的节约量：

$$
\Delta_{\mathrm{recompute}} = M_{\mathrm{offload}}
$$

节约比例：

$$
\rho_{\mathrm{recompute}} = 1
$$

### 4.2 CA Recompute + 3 LA Prefetch

这个策略里：

- CA 不需要在 CPU 上保存 prefix KV
- 3 个 LA 仍然要从 CPU prefetch 完整 state

因此：

$$
M_{\mathrm{CA\_recompute}}
= B \cdot 3S_{\mathrm{state}}
$$

节约量：

$$
\Delta_{\mathrm{CA\_recompute}}
= M_{\mathrm{offload}} - M_{\mathrm{CA\_recompute}}
= B \cdot PS_{\mathrm{kv}}
$$

节约比例：

$$
\rho_{\mathrm{CA\_recompute}}
= \frac{PS_{\mathrm{kv}}}{3S_{\mathrm{state}} + PS_{\mathrm{kv}}}
$$

代入数值：

$$
\rho_{\mathrm{CA\_recompute}}
= \frac{4096P}{6438912 + 4096P}
= \frac{P}{1572 + P}
$$

### 4.3 LA 前 $a$ 层 Recompute，剩余层 + CA 继续 Onload

这个策略里：

- 前 $a$ 个 LA layer 不需要在 CPU 上保存完整 state
- 剩余 $3-a$ 个 LA layer 仍然需要完整 state
- CA 仍然需要保存 prefix KV

因此：

$$
M_{\mathrm{LA\_recompute}}(a)
= B \cdot ((3-a)S_{\mathrm{state}} + PS_{\mathrm{kv}})
$$

节约量：

$$
\Delta_{\mathrm{LA\_recompute}}(a)
= M_{\mathrm{offload}} - M_{\mathrm{LA\_recompute}}(a)
= B \cdot aS_{\mathrm{state}}
$$

节约比例：

$$
\rho_{\mathrm{LA\_recompute}}(a)
= \frac{aS_{\mathrm{state}}}{3S_{\mathrm{state}} + PS_{\mathrm{kv}}}
$$

代入数值：

$$
\rho_{\mathrm{LA\_recompute}}(a)
= \frac{2146304a}{6438912 + 4096P}
= \frac{524a}{1572 + P}
$$

## 5. 代入默认配置后的结果

下面取最常见的 $B = 1$。

### 5.1 当 $P = 4096$

基线：

$$
M_{\mathrm{offload}}
= 6438912 + 4096 \times 4096
= 23216128 \text{ bytes}
= 22.140625 \text{ MiB}
$$

各策略结果：

- Pure Recompute：
  $$
  M = 0,\quad
  \Delta = 22.140625 \text{ MiB},\quad
  \rho = 100\%
  $$
- CA Recompute + 3 LA Prefetch：
  $$
  M = 6.140625 \text{ MiB},\quad
  \Delta = 16.0 \text{ MiB},\quad
  \rho \approx 72.27\%
  $$
- LA 重计算 $a=1$：
  $$
  M = 20.09375 \text{ MiB},\quad
  \Delta = 2.046875 \text{ MiB},\quad
  \rho \approx 9.24\%
  $$
- LA 重计算 $a=2$：
  $$
  M = 18.046875 \text{ MiB},\quad
  \Delta = 4.09375 \text{ MiB},\quad
  \rho \approx 18.49\%
  $$
- LA 重计算 $a=3$：
  $$
  M = 16.0 \text{ MiB},\quad
  \Delta = 6.140625 \text{ MiB},\quad
  \rho \approx 27.73\%
  $$

### 5.2 当 $P = 8192$

基线：

$$
M_{\mathrm{offload}} = 38.140625 \text{ MiB}
$$

各策略结果：

- Pure Recompute：
  $$
  \Delta = 38.140625 \text{ MiB},\quad \rho = 100\%
  $$
- CA Recompute + 3 LA Prefetch：
  $$
  M = 6.140625 \text{ MiB},\quad
  \Delta = 32.0 \text{ MiB},\quad
  \rho \approx 83.90\%
  $$
- LA 重计算 $a=1$：
  $$
  \Delta = 2.046875 \text{ MiB},\quad \rho \approx 5.37\%
  $$
- LA 重计算 $a=2$：
  $$
  \Delta = 4.09375 \text{ MiB},\quad \rho \approx 10.73\%
  $$
- LA 重计算 $a=3$：
  $$
  \Delta = 6.140625 \text{ MiB},\quad \rho \approx 16.10\%
  $$

### 5.3 当 $P = 16384$

基线：

$$
M_{\mathrm{offload}} = 70.140625 \text{ MiB}
$$

各策略结果：

- Pure Recompute：
  $$
  \Delta = 70.140625 \text{ MiB},\quad \rho = 100\%
  $$
- CA Recompute + 3 LA Prefetch：
  $$
  M = 6.140625 \text{ MiB},\quad
  \Delta = 64.0 \text{ MiB},\quad
  \rho \approx 91.25\%
  $$
- LA 重计算 $a=1$：
  $$
  \Delta = 2.046875 \text{ MiB},\quad \rho \approx 2.92\%
  $$
- LA 重计算 $a=2$：
  $$
  \Delta = 4.09375 \text{ MiB},\quad \rho \approx 5.84\%
  $$
- LA 重计算 $a=3$：
  $$
  \Delta = 6.140625 \text{ MiB},\quad \rho \approx 8.75\%
  $$

## 6. 结论

现在 LA prefetch 已经是完整的 `conv + temporal` state，因此：

- `CA recompute + 3 LA prefetch` 节省的是整块 CA prefix KV
- `LA 重计算 a 层` 节省的是 $a$ 层完整 LA state

在当前默认配置下：

- 单层 LA state 约为 `2.046875 MiB`
- 3 层 LA state 合计约为 `6.140625 MiB`
- 单个 CA prefix token 的 KV 为 `4 KiB`

因此当 prefix 很长时，CA KV 仍然会成为更大的 CPU 内存项；但和之前“只算 conv state”相比，LA 重计算带来的 CPU 内存收益已经明显变大了。
