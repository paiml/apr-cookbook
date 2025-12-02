# apr-info

Display APR model information and metadata.

## Usage

```bash
cargo run --example apr_info -- [OPTIONS] [FILE]
```

## Options

| Option | Description |
|--------|-------------|
| `FILE` | Path to APR model file |
| `--demo` | Use demo mode with sample model |
| `-v, --verbose` | Show verbose output (hex dump) |
| `-h, --help` | Print help |
| `-V, --version` | Print version |

## Examples

```bash
# Inspect a model file
cargo run --example apr_info -- model.apr

# Demo mode
cargo run --example apr_info -- --demo

# Verbose output
cargo run --example apr_info -- --verbose model.apr
```

## Output

```
=== APR Model Info ===

Name:        sentiment-classifier
Size:        1048576 bytes
Version:     1.0

Flags:
  Compressed: yes
  Encrypted:  no
  Signed:     no
```

## Verbose Output

With `--verbose`, includes hex dump of header:

```
Header (hex):
  0000: 41 50 52 4e 01 00 01 00 00 00 00 00 00 10 00 00
  0010: 73 65 6e 74 69 6d 65 6e 74 2d 63 6c 61 73 73 69
```
