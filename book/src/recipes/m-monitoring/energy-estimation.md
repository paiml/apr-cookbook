# RAPL Energy Estimation

Estimate energy consumption (joules/inference) using Intel RAPL or TDP-based fallback. Measures per-workload energy, converts to CO2 grams using US grid average, and produces a JSON efficiency report.

**Device**: ![x86_64](https://img.shields.io/badge/-x86__64-blue)

```bash
cargo run --example monitoring_energy_estimation
```

**Key concepts**: Intel RAPL interface, TDP-based estimation fallback, joules-to-CO2 conversion, Green AI metrics.
