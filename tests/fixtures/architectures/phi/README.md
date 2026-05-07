# Phi Synthetic Micro-Fixture (PMAT-304)

2-layer Phi-3 config. Discriminator: qkv_proj_fused=true (Phi-3 fuses
Q/K/V into a single projection, vs Llama's separate q_proj/k_proj/v_proj).

Companion converter `examples/conversion/convert_phi_to_apr.rs` already
exists from earlier work and handles the qkv-split during HF→APR.
