//! # SIMD Software Prefetch Distance Picker
//!
//! Streaming SIMD inference benefits from `prefetch` instructions
//! issued some lookahead bytes ahead of the current load. Distance:
//! too short = miss not hidden; too far = pollutes cache. Heuristic:
//! distance = mem_latency_cycles × bytes_per_cycle, rounded to a
//! cache-line multiple. This recipe builds the picker.
//!
//! Demonstrates the **SIMD.10** recipe for PMAT-134 (simd coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Intel optimization manual § software prefetch tuning.
//!
//! Run with: cargo run --example simd_prefetch_distance_picker
//!
//! Added by PMAT-134 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const CACHE_LINE_BYTES: u32 = 64;
const MAX_DISTANCE_BYTES: u32 = 8 * 1024;

#[derive(Debug, PartialEq)]
pub enum PrefetchVerdict {
    Ok {
        distance_bytes: u32,
        cache_lines: u32,
    },
    InvalidLatency,
    InvalidBandwidth,
    AboveMaxDistance {
        recommended: u32,
    },
}

pub fn pick(mem_latency_cycles: u32, bytes_per_cycle: u32) -> PrefetchVerdict {
    if mem_latency_cycles == 0 {
        return PrefetchVerdict::InvalidLatency;
    }
    if bytes_per_cycle == 0 {
        return PrefetchVerdict::InvalidBandwidth;
    }
    let raw = mem_latency_cycles.saturating_mul(bytes_per_cycle);
    let aligned = raw.div_ceil(CACHE_LINE_BYTES) * CACHE_LINE_BYTES;
    if aligned > MAX_DISTANCE_BYTES {
        return PrefetchVerdict::AboveMaxDistance {
            recommended: MAX_DISTANCE_BYTES,
        };
    }
    PrefetchVerdict::Ok {
        distance_bytes: aligned,
        cache_lines: aligned / CACHE_LINE_BYTES,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("simd_prefetch_distance_picker")?;

    let cases = [
        (200u32, 16u32), // 3200 → 3200 (50 cache lines)
        (50, 16),        // 800 → 832 (13 cache lines)
        (200, 100),      // 20000 → capped
        (0, 16),
        (200, 0),
    ];
    for (lat, bw) in cases {
        println!("lat={lat}c bw={bw}b/c → {:?}", pick(lat, bw));
    }
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
    fn typical_distance_aligned_to_cache_line() {
        // 50 cycles × 16 bytes/cycle = 800 → ceil to 64-byte boundary = 832.
        let v = pick(50, 16);
        assert!(matches!(
            v,
            PrefetchVerdict::Ok {
                distance_bytes: 832,
                cache_lines: 13
            }
        ));
    }

    #[test]
    fn already_aligned_passes_through() {
        // 200 × 16 = 3200, exactly 50 cache lines.
        let v = pick(200, 16);
        assert!(matches!(
            v,
            PrefetchVerdict::Ok {
                distance_bytes: 3200,
                cache_lines: 50
            }
        ));
    }

    #[test]
    fn excessive_distance_capped() {
        let v = pick(1000, 100);
        assert!(matches!(v, PrefetchVerdict::AboveMaxDistance { .. }));
    }

    #[test]
    fn zero_latency_invalid() {
        assert_eq!(pick(0, 16), PrefetchVerdict::InvalidLatency);
    }

    #[test]
    fn zero_bandwidth_invalid() {
        assert_eq!(pick(50, 0), PrefetchVerdict::InvalidBandwidth);
    }

    #[test]
    fn small_latency_min_distance_one_line() {
        // 1 × 1 = 1 byte → ceil to 64 bytes = 1 cache line.
        let v = pick(1, 1);
        assert!(matches!(
            v,
            PrefetchVerdict::Ok {
                distance_bytes: 64,
                cache_lines: 1
            }
        ));
    }

    #[test]
    fn distance_always_multiple_of_64() {
        for (lat, bw) in [(7u32, 5u32), (13, 11), (97, 3)] {
            if let PrefetchVerdict::Ok { distance_bytes, .. } = pick(lat, bw) {
                assert_eq!(
                    distance_bytes % CACHE_LINE_BYTES,
                    0,
                    "distance not aligned for ({lat},{bw})"
                );
            }
        }
    }

    #[test]
    fn cache_lines_match_distance() {
        if let PrefetchVerdict::Ok {
            distance_bytes,
            cache_lines,
        } = pick(100, 8)
        {
            assert_eq!(cache_lines, distance_bytes / CACHE_LINE_BYTES);
        }
    }

    #[test]
    fn at_max_passes() {
        // 8192 / 16 = 512 cycles → exactly MAX_DISTANCE_BYTES.
        let v = pick(512, 16);
        assert!(matches!(
            v,
            PrefetchVerdict::Ok {
                distance_bytes: 8192,
                ..
            }
        ));
    }

    #[test]
    fn just_above_max_capped() {
        // 8193 → ceil 8256 (129 lines) > MAX.
        let v = pick(513, 16);
        assert!(matches!(v, PrefetchVerdict::AboveMaxDistance { .. }));
    }
}
