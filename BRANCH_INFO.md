# Branch: phase3c-v2-direct-write

This is the **mlx-lm model code** for Mistral Small 4 with absorbed MLA + INT4
quantized latent cache + fused Metal SDPA kernel.

## Companion repo

This branch requires the fused SDPA Metal kernel from the MLX fork:

**`ProducerGuy/mlx` @ `phase3c-v3-kernel-opt`**

The branch names don't match (`v2` here, `v3` there) but they are the same
release — one package is the model code (this repo), the other is the kernel
(MLX fork). They must be installed together.

## Install (both required)

```bash
pip install git+https://github.com/ProducerGuy/mlx.git@phase3c-v3-kernel-opt
pip install git+https://github.com/ProducerGuy/mlx-lm.git@phase3c-v2-direct-write
```

## What this branch contains

- `mlx_lm/models/mistral4.py` — Absorbed MLA + fused decode path + MoEGate fix
- `mlx_lm/models/mistral3.py` — Model routing
- `mlx_lm/models/cache.py` — `QuantizedLatentKVCache` with direct cache update

## PRs

- ml-explore/mlx-lm#1037 (this model code)
- ml-explore/mlx#3373 (the companion Metal kernel)
