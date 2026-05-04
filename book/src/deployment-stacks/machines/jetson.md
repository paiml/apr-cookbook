# Jetson Edge Machine

NVIDIA Jetson provisioning for edge inference.

## Files

- **Canary deployment**: [`examples/machines/jetson/canary/`](https://github.com/paiml/apr-cookbook/tree/main/examples/machines/jetson/canary)
- **Makefile**: [`examples/machines/jetson/Makefile`](https://github.com/paiml/apr-cookbook/blob/main/examples/machines/jetson/Makefile)

## Usage

```bash
cd examples/machines/jetson
make help
```

## Companion recipes

- `jetson-edge-base.yaml` -- base image provisioning
- Stacks `09-edge-inference` -- full edge inference deployment

## Provenance

Migrated from `sovereign-ai-cookbook/machines/jetson/` by PMAT-065 (centralize-cookbooks).
