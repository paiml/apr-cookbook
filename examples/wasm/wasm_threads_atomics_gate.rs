//! # WASM Threads + Atomics Capability Gate
//!
//! Multi-threaded WASM requires three browser features simultaneously:
//! `WebAssembly.Memory({ shared: true })`, the `atomics` proposal, and
//! cross-origin isolation (`COOP+COEP` headers). Missing any → fall
//! back to single-threaded. This recipe builds the gate.
//!
//! Demonstrates the **WASM.11** recipe for PMAT-134 (wasm coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly threads proposal; W3C cross-origin isolation spec.
//!
//! Run with: cargo run --example wasm_threads_atomics_gate
//!
//! Added by PMAT-134 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadsCapability {
    pub shared_memory: bool,
    pub atomics: bool,
    pub cross_origin_isolated: bool,
}

#[derive(Debug, PartialEq)]
pub enum ThreadsVerdict {
    Enabled { recommended_workers: u32 },
    Disabled { missing: Vec<&'static str> },
    InvalidWorkerCount,
}

pub fn decide(cap: ThreadsCapability, hardware_concurrency: u32) -> ThreadsVerdict {
    if hardware_concurrency == 0 {
        return ThreadsVerdict::InvalidWorkerCount;
    }
    let mut missing: Vec<&'static str> = Vec::new();
    if !cap.shared_memory {
        missing.push("shared_memory");
    }
    if !cap.atomics {
        missing.push("atomics");
    }
    if !cap.cross_origin_isolated {
        missing.push("cross_origin_isolated");
    }
    if !missing.is_empty() {
        return ThreadsVerdict::Disabled { missing };
    }
    let recommended = hardware_concurrency.saturating_sub(1).max(1);
    ThreadsVerdict::Enabled {
        recommended_workers: recommended,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_threads_atomics_gate")?;

    let full = ThreadsCapability {
        shared_memory: true,
        atomics: true,
        cross_origin_isolated: true,
    };
    println!("full caps, 8 cores: {:?}", decide(full, 8));

    let no_coop = ThreadsCapability {
        cross_origin_isolated: false,
        ..full
    };
    println!("missing COOP: {:?}", decide(no_coop, 8));

    let nothing = ThreadsCapability {
        shared_memory: false,
        atomics: false,
        cross_origin_isolated: false,
    };
    println!("nothing: {:?}", decide(nothing, 8));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_runs() {
        main().expect("recipe execution failed");
    }

    fn full() -> ThreadsCapability {
        ThreadsCapability {
            shared_memory: true,
            atomics: true,
            cross_origin_isolated: true,
        }
    }

    #[test]
    fn full_caps_enables_threads() {
        let v = decide(full(), 8);
        assert!(matches!(
            v,
            ThreadsVerdict::Enabled {
                recommended_workers: 7
            }
        ));
    }

    #[test]
    fn missing_shared_memory_disabled() {
        let cap = ThreadsCapability {
            shared_memory: false,
            ..full()
        };
        if let ThreadsVerdict::Disabled { missing } = decide(cap, 8) {
            assert!(missing.contains(&"shared_memory"));
        }
    }

    #[test]
    fn missing_atomics_disabled() {
        let cap = ThreadsCapability {
            atomics: false,
            ..full()
        };
        if let ThreadsVerdict::Disabled { missing } = decide(cap, 8) {
            assert!(missing.contains(&"atomics"));
        }
    }

    #[test]
    fn missing_coop_disabled() {
        let cap = ThreadsCapability {
            cross_origin_isolated: false,
            ..full()
        };
        if let ThreadsVerdict::Disabled { missing } = decide(cap, 8) {
            assert!(missing.contains(&"cross_origin_isolated"));
        }
    }

    #[test]
    fn nothing_lists_all_missing() {
        let cap = ThreadsCapability {
            shared_memory: false,
            atomics: false,
            cross_origin_isolated: false,
        };
        if let ThreadsVerdict::Disabled { missing } = decide(cap, 8) {
            assert_eq!(missing.len(), 3);
        }
    }

    #[test]
    fn zero_concurrency_invalid() {
        assert_eq!(decide(full(), 0), ThreadsVerdict::InvalidWorkerCount);
    }

    #[test]
    fn single_core_recommends_one_worker() {
        // 1 - 1 = 0, but we floor at 1.
        let v = decide(full(), 1);
        assert!(matches!(
            v,
            ThreadsVerdict::Enabled {
                recommended_workers: 1
            }
        ));
    }

    #[test]
    fn high_concurrency_recommends_n_minus_one() {
        let v = decide(full(), 32);
        assert!(matches!(
            v,
            ThreadsVerdict::Enabled {
                recommended_workers: 31
            }
        ));
    }

    #[test]
    fn dual_core_recommends_one_worker() {
        let v = decide(full(), 2);
        assert!(matches!(
            v,
            ThreadsVerdict::Enabled {
                recommended_workers: 1
            }
        ));
    }
}
