//! # WASM Module Cache Strategy Picker
//!
//! Compiling a WASM module is expensive (~ 100 ms / MiB on M1). Caching
//! options: in-memory (fast, lost on tab close), IndexedDB (persistent
//! across sessions), Cache API (HTTP-backed, sw-friendly), none
//! (always recompile). This recipe picks the strategy by size + use
//! case.
//!
//! Demonstrates the **WASM.8** recipe for PMAT-123 (wasm coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WICG WebAssembly Compile-Streams + Cache API conventions
//!
//! Run with: cargo run --example wasm_module_cache_strategy
//!
//! Added by PMAT-123 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStrategy {
    None,
    InMemory,
    IndexedDb,
    CacheApi,
}

#[derive(Debug, PartialEq)]
pub enum PickerVerdict {
    Ok(CacheStrategy),
    InvalidSize,
}

const TINY_MIB: u32 = 1;
const SMALL_MIB: u32 = 10;
const MEDIUM_MIB: u32 = 100;

pub fn pick(module_size_mib: u32, persistent: bool, http_backed: bool) -> PickerVerdict {
    if module_size_mib == 0 {
        return PickerVerdict::InvalidSize;
    }
    // For tiny modules, recompile beats cache lookup overhead.
    if module_size_mib <= TINY_MIB && !persistent {
        return PickerVerdict::Ok(CacheStrategy::None);
    }
    if http_backed {
        return PickerVerdict::Ok(CacheStrategy::CacheApi);
    }
    if persistent {
        return PickerVerdict::Ok(CacheStrategy::IndexedDb);
    }
    if module_size_mib <= SMALL_MIB {
        PickerVerdict::Ok(CacheStrategy::InMemory)
    } else if module_size_mib <= MEDIUM_MIB {
        PickerVerdict::Ok(CacheStrategy::IndexedDb)
    } else {
        // Large modules: persist via IndexedDB regardless.
        PickerVerdict::Ok(CacheStrategy::IndexedDb)
    }
}

pub fn estimated_compile_ms(module_size_mib: u32) -> u64 {
    // Rough heuristic: 100 ms / MiB.
    u64::from(module_size_mib) * 100
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_module_cache_strategy")?;

    let cases = [
        (1, false, false),
        (5, false, false),
        (50, false, false),
        (200, true, false),
        (5, false, true),
    ];
    for (mib, persistent, http) in cases {
        println!(
            "{mib} MiB persistent={persistent} http={http}  →  {:?}",
            pick(mib, persistent, http)
        );
    }
    println!("compile(50 MiB) = {} ms", estimated_compile_ms(50));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn tiny_non_persistent_uses_none() {
        assert_eq!(
            pick(1, false, false),
            PickerVerdict::Ok(CacheStrategy::None)
        );
    }

    #[test]
    fn small_non_persistent_uses_in_memory() {
        assert_eq!(
            pick(5, false, false),
            PickerVerdict::Ok(CacheStrategy::InMemory)
        );
    }

    #[test]
    fn medium_non_persistent_uses_indexeddb() {
        assert_eq!(
            pick(50, false, false),
            PickerVerdict::Ok(CacheStrategy::IndexedDb)
        );
    }

    #[test]
    fn large_uses_indexeddb_for_persistence() {
        assert_eq!(
            pick(200, false, false),
            PickerVerdict::Ok(CacheStrategy::IndexedDb)
        );
    }

    #[test]
    fn http_backed_uses_cache_api() {
        // HTTP source → service-worker-friendly Cache API.
        assert_eq!(
            pick(5, false, true),
            PickerVerdict::Ok(CacheStrategy::CacheApi)
        );
    }

    #[test]
    fn explicit_persistent_overrides_in_memory() {
        // 5 MiB normally goes in-memory, but if persistent requested → IndexedDB.
        assert_eq!(
            pick(5, true, false),
            PickerVerdict::Ok(CacheStrategy::IndexedDb)
        );
    }

    #[test]
    fn zero_size_invalid() {
        assert_eq!(pick(0, false, false), PickerVerdict::InvalidSize);
    }

    #[test]
    fn compile_estimate_scales_with_size() {
        assert_eq!(estimated_compile_ms(1), 100);
        assert_eq!(estimated_compile_ms(10), 1000);
    }

    #[test]
    fn http_takes_priority_over_persistent() {
        // HTTP-backed is the strongest signal.
        assert_eq!(
            pick(50, true, true),
            PickerVerdict::Ok(CacheStrategy::CacheApi)
        );
    }
}
