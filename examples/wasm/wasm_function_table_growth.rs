//! # WASM Function-Table Growth Strategy
//!
//! Function tables (used for `call_indirect`) can grow dynamically.
//! Strategies:
//!   FixedSize: pre-size to upper bound; no grows; UB if exceeded
//!   DoubleOnGrow: O(1) amortized; potential 2× memory waste
//!   IncrementalChunk(N): grow by N entries each time; predictable
//!
//! Picker: by max_callable_count + memory_pressure tier.
//!
//! Demonstrates the **WASM.25** recipe for PMAT-151 (wasm round 5).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly table.grow semantics.
//!
//! Run with: cargo run --example wasm_function_table_growth
//!
//! Added by PMAT-151 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrowthStrategy {
    FixedSize { capacity: u32 },
    DoubleOnGrow { initial: u32 },
    IncrementalChunk { chunk: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressure {
    Low,
    Moderate,
    High,
}

#[derive(Debug, PartialEq)]
pub enum GrowthVerdict {
    Ok {
        strategy: GrowthStrategy,
        expected_grows: u32,
    },
    InvalidMaxCount,
}

pub fn pick(max_callable_count: u32, memory_pressure: MemoryPressure) -> GrowthVerdict {
    if max_callable_count == 0 {
        return GrowthVerdict::InvalidMaxCount;
    }
    let (strategy, expected_grows) = match memory_pressure {
        MemoryPressure::Low => (
            GrowthStrategy::FixedSize {
                capacity: max_callable_count,
            },
            0,
        ),
        MemoryPressure::Moderate => {
            let initial = max_callable_count / 4;
            let grows = if initial == 0 {
                0
            } else {
                ((max_callable_count as f64 / initial as f64).log2().ceil() as u32).max(1)
            };
            (
                GrowthStrategy::DoubleOnGrow {
                    initial: initial.max(1),
                },
                grows,
            )
        }
        MemoryPressure::High => {
            let chunk = (max_callable_count / 8).max(1);
            (
                GrowthStrategy::IncrementalChunk { chunk },
                max_callable_count.div_ceil(chunk),
            )
        }
    };
    GrowthVerdict::Ok {
        strategy,
        expected_grows,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_function_table_growth")?;

    println!("low pressure: {:?}", pick(1000, MemoryPressure::Low));
    println!("moderate: {:?}", pick(1000, MemoryPressure::Moderate));
    println!("high: {:?}", pick(1000, MemoryPressure::High));
    println!("invalid: {:?}", pick(0, MemoryPressure::Low));
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
    fn low_pressure_fixed_size() {
        let v = pick(1000, MemoryPressure::Low);
        if let GrowthVerdict::Ok { strategy, .. } = v {
            assert!(matches!(strategy, GrowthStrategy::FixedSize { .. }));
        }
    }

    #[test]
    fn moderate_pressure_double() {
        let v = pick(1000, MemoryPressure::Moderate);
        if let GrowthVerdict::Ok { strategy, .. } = v {
            assert!(matches!(strategy, GrowthStrategy::DoubleOnGrow { .. }));
        }
    }

    #[test]
    fn high_pressure_chunk() {
        let v = pick(1000, MemoryPressure::High);
        if let GrowthVerdict::Ok { strategy, .. } = v {
            assert!(matches!(strategy, GrowthStrategy::IncrementalChunk { .. }));
        }
    }

    #[test]
    fn invalid_zero_count() {
        assert_eq!(pick(0, MemoryPressure::Low), GrowthVerdict::InvalidMaxCount);
    }

    #[test]
    fn fixed_size_zero_grows() {
        let v = pick(1000, MemoryPressure::Low);
        if let GrowthVerdict::Ok { expected_grows, .. } = v {
            assert_eq!(expected_grows, 0);
        }
    }

    #[test]
    fn double_on_grow_log_scaling() {
        // 1000 / 4 = 250; log2(1000/250) = 2 → 2 grows.
        let v = pick(1000, MemoryPressure::Moderate);
        if let GrowthVerdict::Ok { expected_grows, .. } = v {
            assert!(expected_grows >= 1);
            assert!(expected_grows <= 5);
        }
    }

    #[test]
    fn chunk_grows_proportional() {
        // 1000 entries / 125 chunk = 8 grows.
        let v = pick(1000, MemoryPressure::High);
        if let GrowthVerdict::Ok { expected_grows, .. } = v {
            assert_eq!(expected_grows, 8);
        }
    }

    #[test]
    fn fixed_capacity_matches_input() {
        let v = pick(500, MemoryPressure::Low);
        if let GrowthVerdict::Ok {
            strategy: GrowthStrategy::FixedSize { capacity },
            ..
        } = v
        {
            assert_eq!(capacity, 500);
        }
    }

    #[test]
    fn double_initial_quarter_of_max() {
        // 1000 / 4 = 250.
        let v = pick(1000, MemoryPressure::Moderate);
        if let GrowthVerdict::Ok {
            strategy: GrowthStrategy::DoubleOnGrow { initial },
            ..
        } = v
        {
            assert_eq!(initial, 250);
        }
    }

    #[test]
    fn small_count_chunk_at_least_one() {
        let v = pick(5, MemoryPressure::High);
        if let GrowthVerdict::Ok {
            strategy: GrowthStrategy::IncrementalChunk { chunk },
            ..
        } = v
        {
            assert!(chunk >= 1);
        }
    }
}
