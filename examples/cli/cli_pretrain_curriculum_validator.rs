//! # apr pretrain --curriculum — Stage Schedule Validator
//!
//! Curriculum learning ramps difficulty: short sequences first, longer
//! later. Stages: `(seq_len, frac_of_total_steps)` tuples; constraints:
//! seq_len monotonically non-decreasing; fractions sum to 1.0; seq_len
//! ≤ context_window. This recipe builds the validator.
//!
//! Demonstrates the **PRE.4** recipe for PMAT-117 (apr pretrain coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PRE-001 + Bengio et al. 2009 (curriculum learning)
//!
//! Run with: cargo run --example cli_pretrain_curriculum_validator
//!
//! Added by PMAT-117 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const SUM_TOLERANCE: f64 = 1e-6;

#[derive(Debug, Clone, Copy)]
pub struct Stage {
    pub seq_len: u32,
    pub fraction: f64,
}

#[derive(Debug, PartialEq)]
pub enum CurriculumVerdict {
    Ok,
    EmptyCurriculum,
    SeqLenDecreased {
        at_index: usize,
        prev: u32,
        curr: u32,
    },
    SeqLenExceedsContext {
        at_index: usize,
        seq_len: u32,
        ctx: u32,
    },
    FractionsDoNotSumToOne {
        sum: f64,
    },
    InvalidFraction {
        at_index: usize,
        value: f64,
    },
    ZeroSeqLen {
        at_index: usize,
    },
}

pub fn validate(stages: &[Stage], context_window: u32) -> CurriculumVerdict {
    if stages.is_empty() {
        return CurriculumVerdict::EmptyCurriculum;
    }
    let mut sum = 0.0;
    let mut prev_seq = 0u32;
    for (i, stage) in stages.iter().enumerate() {
        if stage.seq_len == 0 {
            return CurriculumVerdict::ZeroSeqLen { at_index: i };
        }
        if stage.seq_len > context_window {
            return CurriculumVerdict::SeqLenExceedsContext {
                at_index: i,
                seq_len: stage.seq_len,
                ctx: context_window,
            };
        }
        if i > 0 && stage.seq_len < prev_seq {
            return CurriculumVerdict::SeqLenDecreased {
                at_index: i,
                prev: prev_seq,
                curr: stage.seq_len,
            };
        }
        if !stage.fraction.is_finite() || stage.fraction <= 0.0 || stage.fraction > 1.0 {
            return CurriculumVerdict::InvalidFraction {
                at_index: i,
                value: stage.fraction,
            };
        }
        sum += stage.fraction;
        prev_seq = stage.seq_len;
    }
    if (sum - 1.0).abs() > SUM_TOLERANCE {
        return CurriculumVerdict::FractionsDoNotSumToOne { sum };
    }
    CurriculumVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_pretrain_curriculum_validator")?;

    let valid = vec![
        Stage {
            seq_len: 512,
            fraction: 0.3,
        },
        Stage {
            seq_len: 1024,
            fraction: 0.4,
        },
        Stage {
            seq_len: 2048,
            fraction: 0.3,
        },
    ];
    println!("valid (ctx=8K):  {:?}", validate(&valid, 8192));

    let bad_order = vec![
        Stage {
            seq_len: 2048,
            fraction: 0.5,
        },
        Stage {
            seq_len: 512,
            fraction: 0.5,
        },
    ];
    println!("bad order:       {:?}", validate(&bad_order, 8192));

    let bad_sum = vec![
        Stage {
            seq_len: 512,
            fraction: 0.3,
        },
        Stage {
            seq_len: 1024,
            fraction: 0.3,
        },
    ];
    println!("bad sum (0.6):   {:?}", validate(&bad_sum, 8192));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_curriculum_passes() {
        let stages = vec![
            Stage {
                seq_len: 512,
                fraction: 0.3,
            },
            Stage {
                seq_len: 1024,
                fraction: 0.4,
            },
            Stage {
                seq_len: 2048,
                fraction: 0.3,
            },
        ];
        assert_eq!(validate(&stages, 8192), CurriculumVerdict::Ok);
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(validate(&[], 8192), CurriculumVerdict::EmptyCurriculum);
    }

    #[test]
    fn zero_seq_rejected() {
        let s = vec![Stage {
            seq_len: 0,
            fraction: 1.0,
        }];
        let v = validate(&s, 8192);
        assert!(matches!(v, CurriculumVerdict::ZeroSeqLen { at_index: 0 }));
    }

    #[test]
    fn seq_len_decrease_rejected() {
        let s = vec![
            Stage {
                seq_len: 2048,
                fraction: 0.5,
            },
            Stage {
                seq_len: 512,
                fraction: 0.5,
            },
        ];
        let v = validate(&s, 8192);
        assert!(matches!(v, CurriculumVerdict::SeqLenDecreased { .. }));
    }

    #[test]
    fn seq_len_exceeds_ctx_rejected() {
        let s = vec![Stage {
            seq_len: 9000,
            fraction: 1.0,
        }];
        let v = validate(&s, 8192);
        assert!(matches!(v, CurriculumVerdict::SeqLenExceedsContext { .. }));
    }

    #[test]
    fn fractions_short_of_one_rejected() {
        let s = vec![
            Stage {
                seq_len: 512,
                fraction: 0.3,
            },
            Stage {
                seq_len: 1024,
                fraction: 0.3,
            },
        ];
        let v = validate(&s, 8192);
        assert!(matches!(
            v,
            CurriculumVerdict::FractionsDoNotSumToOne { .. }
        ));
    }

    #[test]
    fn invalid_fraction_rejected() {
        let s = vec![Stage {
            seq_len: 512,
            fraction: 1.5,
        }];
        let v = validate(&s, 8192);
        assert!(matches!(v, CurriculumVerdict::InvalidFraction { .. }));
    }

    #[test]
    fn equal_seq_len_passes() {
        // Non-decreasing means equal is OK.
        let s = vec![
            Stage {
                seq_len: 512,
                fraction: 0.5,
            },
            Stage {
                seq_len: 512,
                fraction: 0.5,
            },
        ];
        assert_eq!(validate(&s, 8192), CurriculumVerdict::Ok);
    }

    #[test]
    fn at_ctx_boundary_passes() {
        let s = vec![Stage {
            seq_len: 8192,
            fraction: 1.0,
        }];
        assert_eq!(validate(&s, 8192), CurriculumVerdict::Ok);
    }
}
