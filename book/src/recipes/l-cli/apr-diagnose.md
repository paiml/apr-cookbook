# apr-diagnose

Automated Five Whys root-cause analysis on training checkpoints. Detects symptoms (high loss, NaN gradients, slow convergence, memory spikes, overfitting) and traces through a 5-level diagnostic chain to the root cause.

```bash
cargo run --example cli_apr_diagnose -- --demo
```

**CLI equivalent**: `apr diagnose checkpoint_epoch_10.apr`
