# Memory Profiler

Track peak RSS during model load and inference via `/proc/self/status`. Profiles multiple model sizes, computes peak memory delta, and generates container/Lambda sizing recommendations.

**Device**: ![cpu](https://img.shields.io/badge/-cpu-lightgrey)

```bash
cargo run --example monitoring_memory_profiler
```

**Key concepts**: RSS tracking via procfs, phase-based memory profiling, Lambda/Docker sizing recommendations, headroom calculation.
