# apr-ptx-map

Model-to-PTX source mapping for GPU kernel visibility (Mieruka principle). Maps transformer layers to PTX kernels, computes theoretical SM occupancy from register pressure and shared memory, and shows instruction category breakdown.

```bash
cargo run --example cli_apr_ptx_map -- --demo
```

**CLI equivalent**: `apr ptx-map model.apr`, `apr ptx-map --kernel-filter attention model.apr`
