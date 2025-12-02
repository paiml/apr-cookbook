# WASM Deployment

Deploy APR models to browsers and edge functions.

## Building for WASM

```bash
# Install target
rustup target add wasm32-unknown-unknown

# Build
cargo build --target wasm32-unknown-unknown --release
```

## WASM-Compatible Code

```rust
use apr_cookbook::bundle::BundledModel;

// Works in WASM - no file I/O
const MODEL: &[u8] = include_bytes!("model.apr");

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub fn load_model() -> Result<JsValue, JsValue> {
    let model = BundledModel::from_bytes(MODEL)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(JsValue::from_str(model.name()))
}
```

## JavaScript Usage

```javascript
import init, { load_model } from './pkg/my_model.js';

async function main() {
    await init();
    const name = load_model();
    console.log(`Loaded model: ${name}`);
}

main();
```

## Size Optimization

| Technique | Size Reduction |
|-----------|----------------|
| `--release` | 50-70% |
| `opt-level = 'z'` | 10-20% |
| `wasm-opt -Oz` | 5-15% |
| `lto = true` | 10-20% |

## Cargo.toml for WASM

```toml
[profile.release]
opt-level = 'z'
lto = true
codegen-units = 1
panic = 'abort'
```
