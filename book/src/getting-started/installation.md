# Installation

## Requirements

- Rust 1.75 or later
- Cargo (included with Rust)

## Add to Cargo.toml

```toml
[dependencies]
apr-cookbook = "0.1"
```

## Feature Flags

Enable optional features as needed:

```toml
[dependencies]
apr-cookbook = { version = "0.1", features = ["encryption"] }
```

| Feature | Description |
|---------|-------------|
| `default` | Core bundling and conversion |
| `encryption` | AES-256-GCM model encryption |
| `training` | Integration with entrenar |
| `full` | All features enabled |

## Verify Installation

```rust
use apr_cookbook::bundle::ModelBundle;

fn main() {
    let bundle = ModelBundle::new()
        .with_name("test")
        .build();

    println!("APR magic: {:?}", &bundle[0..4]);
    // Output: APR magic: [65, 80, 82, 78] (APRN)
}
```

## Development Setup

For contributors:

```bash
git clone https://github.com/paiml/apr-cookbook
cd apr-cookbook
make test-fast    # Run tests
make lint         # Check code quality
make coverage     # Generate coverage report
```
