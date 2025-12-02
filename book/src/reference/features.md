# Feature Flags

Configure apr-cookbook capabilities via Cargo features.

## Available Features

| Feature | Description | Default |
|---------|-------------|---------|
| `default` | Core bundling and conversion | ✅ |
| `encryption` | AES-256-GCM encryption | ❌ |
| `training` | entrenar integration | ❌ |
| `full` | All features | ❌ |

## Usage

### Single Feature

```toml
[dependencies]
apr-cookbook = { version = "0.1", features = ["encryption"] }
```

### Multiple Features

```toml
[dependencies]
apr-cookbook = { version = "0.1", features = ["encryption", "training"] }
```

### All Features

```toml
[dependencies]
apr-cookbook = { version = "0.1", features = ["full"] }
```

## Feature Details

### `encryption`

Enables model encryption with AES-256-GCM:

```rust
#[cfg(feature = "encryption")]
use aprender::format::{save_encrypted, load_encrypted};
```

Adds dependencies:
- `aprender/format-encryption`

### `training`

Enables training integration with entrenar:

```rust
#[cfg(feature = "training")]
use entrenar::Trainer;
```

Adds dependencies:
- `entrenar`

## Checking Features at Runtime

```rust
#[cfg(feature = "encryption")]
fn encrypt_available() -> bool { true }

#[cfg(not(feature = "encryption"))]
fn encrypt_available() -> bool { false }
```

## Conditional Compilation

```rust
pub fn save_model(model: &Model, path: &str, encrypt: bool) -> Result<()> {
    if encrypt {
        #[cfg(feature = "encryption")]
        {
            return save_encrypted(model, path, "password");
        }

        #[cfg(not(feature = "encryption"))]
        {
            return Err(CookbookError::feature_not_available("encryption"));
        }
    }

    save(model, path)
}
```
