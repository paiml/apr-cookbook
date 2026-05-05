//! # apr tokenize — Plan-Phase Resource Estimator
//!
//! `apr tokenize plan <CORPUS>` validates inputs and estimates training
//! time / RAM for the BPE training that will follow. This recipe builds
//! the estimator as a pure function of (corpus_bytes, vocab_size, threads)
//! so a CI pipeline can preview cost without running the full plan phase.
//!
//! Demonstrates the **TOKENIZE.3** recipe for PMAT-095 (apr tokenize coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender contracts/tokenizer-bpe-v1.yaml + Sennrich et al. (2016) BPE
//!
//! Run with: cargo run --example cli_tokenize_plan_estimator
//!
//! Added by PMAT-095 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct PlanEstimate {
    pub eta_seconds: u64,
    pub peak_ram_bytes: u64,
    pub merges_to_learn: u32,
}

const BASE_BYTES_OVERHEAD: u64 = 256 * 1024 * 1024; // 256 MB of base allocator overhead
const BPE_THROUGHPUT_BYTES_PER_SEC: u64 = 50_000_000; // 50 MB/s per worker thread (ballpark)

pub fn estimate_plan(corpus_bytes: u64, vocab_size: u32, threads: u32) -> Option<PlanEstimate> {
    if corpus_bytes == 0 || vocab_size < 256 || threads == 0 {
        return None;
    }
    let merges = vocab_size.saturating_sub(256);
    // BPE merge passes scale roughly linearly with corpus_bytes × log(vocab).
    let log_vocab = (vocab_size as f64).log2().max(1.0);
    let work = corpus_bytes as f64 * log_vocab;
    let throughput = BPE_THROUGHPUT_BYTES_PER_SEC as f64 * f64::from(threads);
    let eta = (work / throughput).ceil() as u64;
    // Peak RAM: base + 2× corpus bytes (working set) + 64 B per merge entry.
    let ram = BASE_BYTES_OVERHEAD + 2 * corpus_bytes + 64 * u64::from(merges);
    Some(PlanEstimate {
        eta_seconds: eta,
        peak_ram_bytes: ram,
        merges_to_learn: merges,
    })
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_tokenize_plan_estimator")?;

    let cases = [
        ("100 MB / 32k / 8 thr", 100_000_000, 32_000, 8),
        ("1 GB / 50k / 16 thr", 1_000_000_000, 50_000, 16),
        ("10 GB / 100k / 32 thr", 10_000_000_000, 100_000, 32),
        ("zero corpus", 0, 32_000, 8),
        ("vocab too small", 100_000_000, 100, 8),
    ];

    for (label, c, v, t) in cases {
        println!("{label:>22}  →  {:?}", estimate_plan(c, v, t));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_estimate_returns_some() {
        let p = estimate_plan(100_000_000, 32_000, 8).unwrap();
        assert!(p.eta_seconds > 0);
        assert!(p.peak_ram_bytes > 0);
        assert_eq!(p.merges_to_learn, 32_000 - 256);
    }

    #[test]
    fn zero_corpus_returns_none() {
        assert!(estimate_plan(0, 32_000, 8).is_none());
    }

    #[test]
    fn vocab_below_256_returns_none() {
        // BPE starts from 256 byte-level alphabet; vocab < 256 is degenerate.
        assert!(estimate_plan(100_000_000, 100, 8).is_none());
        assert!(estimate_plan(100_000_000, 255, 8).is_none());
    }

    #[test]
    fn vocab_at_256_yields_zero_merges() {
        let p = estimate_plan(100_000_000, 256, 8).unwrap();
        assert_eq!(p.merges_to_learn, 0);
    }

    #[test]
    fn more_threads_reduce_eta() {
        // Linear thread scaling — more workers → shorter ETA.
        let p1 = estimate_plan(1_000_000_000, 32_000, 1).unwrap();
        let p8 = estimate_plan(1_000_000_000, 32_000, 8).unwrap();
        assert!(p8.eta_seconds < p1.eta_seconds);
    }

    #[test]
    fn larger_corpus_increases_ram_proportionally() {
        let small = estimate_plan(100_000_000, 32_000, 8).unwrap();
        let big = estimate_plan(1_000_000_000, 32_000, 8).unwrap();
        // RAM dominated by 2× corpus_bytes term.
        assert!(big.peak_ram_bytes > small.peak_ram_bytes);
    }

    #[test]
    fn zero_threads_returns_none() {
        // No workers = infinite ETA — refuse plan instead of dividing by zero.
        assert!(estimate_plan(100_000_000, 32_000, 0).is_none());
    }
}
