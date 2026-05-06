//! # apr bench — `--warmup` + `--iterations` Budget Validator
//!
//! `apr bench --warmup <W> --iterations <N> --max-tokens <T>` controls
//! the measurement envelope. Budget = W × T (warmup, discarded) + N × T
//! (measured). Need warmup ≥ 1 (avoid cold-cache), iterations ≥ 3 (need
//! 3 samples for percentile statistics), max-tokens ≥ 1.
//!
//! Demonstrates the **BENCH.12** recipe for PMAT-109 (apr bench coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender BENCH-001
//!
//! Run with: cargo run --example cli_bench_warmup_iterations_envelope
//!
//! Added by PMAT-109 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy)]
pub struct BenchEnvelope {
    pub warmup: u32,
    pub iterations: u32,
    pub max_tokens: u32,
}

#[derive(Debug, PartialEq)]
pub enum EnvelopeVerdict {
    Ok {
        warmup_tokens: u64,
        measured_tokens: u64,
    },
    InsufficientWarmup,
    InsufficientIterations,
    ZeroMaxTokens,
}

const MIN_ITERATIONS: u32 = 3;

pub fn validate(env: BenchEnvelope) -> EnvelopeVerdict {
    if env.warmup == 0 {
        return EnvelopeVerdict::InsufficientWarmup;
    }
    if env.iterations < MIN_ITERATIONS {
        return EnvelopeVerdict::InsufficientIterations;
    }
    if env.max_tokens == 0 {
        return EnvelopeVerdict::ZeroMaxTokens;
    }
    let warmup_tokens = u64::from(env.warmup) * u64::from(env.max_tokens);
    let measured = u64::from(env.iterations) * u64::from(env.max_tokens);
    EnvelopeVerdict::Ok {
        warmup_tokens,
        measured_tokens: measured,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_bench_warmup_iterations_envelope")?;

    let cases = [
        (
            "default 3/5/32",
            BenchEnvelope {
                warmup: 3,
                iterations: 5,
                max_tokens: 32,
            },
        ),
        (
            "zero warmup",
            BenchEnvelope {
                warmup: 0,
                iterations: 5,
                max_tokens: 32,
            },
        ),
        (
            "two iterations",
            BenchEnvelope {
                warmup: 1,
                iterations: 2,
                max_tokens: 32,
            },
        ),
        (
            "zero max_tokens",
            BenchEnvelope {
                warmup: 1,
                iterations: 5,
                max_tokens: 0,
            },
        ),
        (
            "strict 5/30/512",
            BenchEnvelope {
                warmup: 5,
                iterations: 30,
                max_tokens: 512,
            },
        ),
    ];
    for (label, env) in cases {
        println!("{label:>20}  →  {:?}", validate(env));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_default_passes() {
        let v = validate(BenchEnvelope {
            warmup: 3,
            iterations: 5,
            max_tokens: 32,
        });
        if let EnvelopeVerdict::Ok {
            warmup_tokens,
            measured_tokens,
        } = v
        {
            assert_eq!(warmup_tokens, 96);
            assert_eq!(measured_tokens, 160);
        }
    }

    #[test]
    fn zero_warmup_rejected() {
        let v = validate(BenchEnvelope {
            warmup: 0,
            iterations: 5,
            max_tokens: 32,
        });
        assert_eq!(v, EnvelopeVerdict::InsufficientWarmup);
    }

    #[test]
    fn iterations_below_three_rejected() {
        // p95/p99 need ≥ 3 samples to be meaningful.
        let v = validate(BenchEnvelope {
            warmup: 1,
            iterations: 2,
            max_tokens: 32,
        });
        assert_eq!(v, EnvelopeVerdict::InsufficientIterations);
    }

    #[test]
    fn boundary_at_three_iterations_passes() {
        let v = validate(BenchEnvelope {
            warmup: 1,
            iterations: 3,
            max_tokens: 32,
        });
        assert!(matches!(v, EnvelopeVerdict::Ok { .. }));
    }

    #[test]
    fn zero_max_tokens_rejected() {
        let v = validate(BenchEnvelope {
            warmup: 1,
            iterations: 5,
            max_tokens: 0,
        });
        assert_eq!(v, EnvelopeVerdict::ZeroMaxTokens);
    }

    #[test]
    fn warmup_tokens_excluded_from_measured() {
        let v = validate(BenchEnvelope {
            warmup: 100,
            iterations: 5,
            max_tokens: 16,
        });
        if let EnvelopeVerdict::Ok {
            warmup_tokens,
            measured_tokens,
        } = v
        {
            // Warmup separately counted; measured = iterations × tokens only.
            assert_eq!(warmup_tokens, 1600);
            assert_eq!(measured_tokens, 80);
        }
    }

    #[test]
    fn very_large_envelope_does_not_overflow() {
        let v = validate(BenchEnvelope {
            warmup: 1,
            iterations: 100,
            max_tokens: u32::MAX,
        });
        assert!(matches!(v, EnvelopeVerdict::Ok { .. }));
    }
}
