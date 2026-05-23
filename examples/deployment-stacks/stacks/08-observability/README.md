# Stack 08: Observability

Monitoring and tracing with **renacer** + Jaeger + Grafana.

## What it deploys

- `renacer` syscall tracer via `cargo install`
- Jaeger all-in-one for distributed tracing (port 16686)
- OTLP collector (port 4317)
- Grafana dashboards (port 3000)
- Firewall rules for all ports

## Usage

```bash
forjar validate -f stacks/08-observability/forjar.yaml
forjar plan -f stacks/08-observability/forjar.yaml
forjar apply -f stacks/08-observability/forjar.yaml
```

## Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `grafana_port` | `3000` | Grafana web UI port |
| `jaeger_port` | `16686` | Jaeger web UI port |
| `retention_days` | `7` | Trace retention period |

## Integration

Point other stack components at this monitor by setting their
OTLP endpoint to `http://<monitor-box>:4317`.
