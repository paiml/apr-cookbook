#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
use apr_cookbook::prelude::*;
use std::fmt;

/// Layer descriptor for synthetic model generation.
pub struct LayerSpec {
    pub name: &'static str,
    pub kernel: &'static str,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_kb: u32,
    pub regs: u32,
}

pub const DEMO_LAYERS: &[LayerSpec] = &[
    LayerSpec {
        name: "embed_tokens",
        kernel: "k_embed_lookup",
        grid: [2048, 1, 1],
        block: [256, 1, 1],
        shared_kb: 0,
        regs: 24,
    },
    LayerSpec {
        name: "layer_0.self_attn",
        kernel: "k_flash_attention_fwd",
        grid: [2048, 32, 1],
        block: [128, 1, 1],
        shared_kb: 48,
        regs: 72,
    },
    LayerSpec {
        name: "layer_0.attn_proj",
        kernel: "k_gemm_nt_fp16",
        grid: [64, 32, 1],
        block: [128, 1, 1],
        shared_kb: 32,
        regs: 48,
    },
    LayerSpec {
        name: "layer_0.mlp.gate",
        kernel: "k_gemm_nt_fp16",
        grid: [86, 32, 1],
        block: [128, 1, 1],
        shared_kb: 32,
        regs: 48,
    },
    LayerSpec {
        name: "layer_0.mlp.up",
        kernel: "k_gemm_nt_fp16",
        grid: [86, 32, 1],
        block: [128, 1, 1],
        shared_kb: 32,
        regs: 48,
    },
    LayerSpec {
        name: "layer_0.mlp.silu",
        kernel: "k_silu_elementwise",
        grid: [2048, 1, 1],
        block: [256, 1, 1],
        shared_kb: 0,
        regs: 16,
    },
    LayerSpec {
        name: "layer_0.mlp.down",
        kernel: "k_gemm_nt_fp16",
        grid: [32, 86, 1],
        block: [128, 1, 1],
        shared_kb: 32,
        regs: 48,
    },
    LayerSpec {
        name: "layer_0.rmsnorm",
        kernel: "k_rmsnorm",
        grid: [2048, 1, 1],
        block: [256, 1, 1],
        shared_kb: 1,
        regs: 20,
    },
    LayerSpec {
        name: "lm_head",
        kernel: "k_gemm_nt_fp16",
        grid: [250, 32, 1],
        block: [128, 1, 1],
        shared_kb: 32,
        regs: 48,
    },
    LayerSpec {
        name: "softmax_sample",
        kernel: "k_softmax_topk",
        grid: [2048, 1, 1],
        block: [256, 1, 1],
        shared_kb: 2,
        regs: 24,
    },
];
