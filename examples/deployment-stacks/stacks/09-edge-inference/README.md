# Stack 09: Edge Inference (Jetson Canary)

Two-machine stack: **intel** cross-compiles, **jetson** runs.

Hardware integration gate for the sovereign AI stack across
CPU NEON, wgpu Vulkan, and CUDA backends.

## Machines

| Machine | Role | Arch | IP |
|---------|------|------|----|
| intel | Builder (cross-compile, CI runner) | x86_64 | 192.168.50.100 |
| jetson | Runner (test execution only) | aarch64 | 192.168.50.53 |

## What it provisions

**Jetson:**
- JetPack 6.2 CUDA stack (nvidia-jetpack apt)
- Sovereign tools (forjar, pmat, pzsh, copia)
- MAXN SUPER power mode
- SSH hardening
- Canary model directory

**Intel:**
- aarch64 cross-compile toolchain (gcc-aarch64-linux-gnu)
- Rust aarch64 targets (stable + nightly)
- SSH config for intel -> jetson

## Usage

```bash
# Provision both machines
forjar apply -f stacks/09-edge-inference/forjar.yaml

# Run full canary pipeline
cd machines/jetson && make canary-all

# Quick verification (post-provision)
cd machines/jetson && make verify
```

## Canary Pipeline

```
Phase 1: apr + model tests
  make canary  (build -> deploy -> check -> qa -> parity -> bench -> chat)

Phase 2: compiled binary + probador
  make canary-e2e  (compile -> serve -> test -> load -> stop)

Full pipeline:
  make canary-all  (both phases)
```

## Spec

See `docs/specifications/sovereign-canary-jetson.md`.
