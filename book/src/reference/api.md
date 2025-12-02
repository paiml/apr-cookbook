# API Documentation

Complete API reference for apr-cookbook.

## Modules

### `apr_cookbook::bundle`

Model bundling and loading.

```rust
pub struct ModelBundle { ... }
pub struct BundledModel<'a> { ... }
```

### `apr_cookbook::convert`

Format conversion utilities.

```rust
pub struct AprConverter { ... }
pub struct TensorData { ... }
pub enum ConversionFormat { ... }
pub enum DataType { ... }
```

### `apr_cookbook::error`

Error types.

```rust
pub enum CookbookError { ... }
pub type Result<T> = std::result::Result<T, CookbookError>;
```

## Full Documentation

Generated API docs are available at:

- [docs.rs/apr-cookbook](https://docs.rs/apr-cookbook)

Or generate locally:

```bash
cargo doc --all-features --open
```

## Stability

| API | Stability |
|-----|-----------|
| `bundle::*` | Stable |
| `convert::*` | Stable |
| `error::*` | Stable |
| `aprender_integration::*` | Experimental |
