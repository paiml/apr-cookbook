# Zero-Copy Loading

Zero-copy loading eliminates memory copies when loading models, reducing latency and memory usage.

## How It Works

Traditional loading:
```
File → Read to buffer → Parse → Copy to model struct → Use
        ↓                        ↓
     Allocation              Allocation
```

Zero-copy loading:
```
Memory (file/include_bytes!) → Interpret in place → Use
                                    ↓
                              No allocations
```

## The `include_bytes!()` Pattern

```rust
// Model bytes are in the binary's .rodata section
const MODEL: &[u8] = include_bytes!("model.apr");

fn main() {
    // BundledModel borrows from MODEL, no copies
    let model = BundledModel::from_bytes(MODEL).unwrap();

    // model.as_bytes() returns the original slice
    assert!(std::ptr::eq(MODEL.as_ptr(), model.as_bytes().as_ptr()));
}
```

## Memory Layout

```
Binary .rodata section:
┌──────────────────────────────────────────┐
│ ... other static data ...                │
│ MODEL: [APRN header | metadata | payload]│
│ ... other static data ...                │
└──────────────────────────────────────────┘
         ↑
         │ BundledModel references this directly
         │ No heap allocations
```

## Benefits

| Metric | Traditional | Zero-Copy |
|--------|-------------|-----------|
| Load time | ~100ms | ~1ms |
| Memory overhead | 2x model size | 0 |
| Allocations | 2+ | 0 |

## When to Use

✅ **Use zero-copy when:**
- Model is embedded via `include_bytes!()`
- Model is memory-mapped
- Model lifetime matches application lifetime

❌ **Don't use when:**
- Model needs modification
- Model comes from untrusted source (validate first)
- Model needs to outlive source buffer
