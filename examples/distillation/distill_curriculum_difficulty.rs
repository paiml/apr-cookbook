//! # Distillation Curriculum Difficulty Picker
//!
//! Order distillation samples easy → hard. Difficulty proxies:
//!   teacher_loss: lower = easier
//!   sample_length: shorter = easier
//!   teacher_confidence: higher = easier
//!
//! Picker sorts samples and returns indices in curriculum order.
//!
//! Demonstrates the **DIST.19** recipe for PMAT-149 (distillation round 5).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Bengio et al. (2009). Curriculum Learning.
//!
//! Run with: cargo run --example distill_curriculum_difficulty
//!
//! Added by PMAT-149 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy)]
pub struct Sample {
    pub teacher_loss: f64,
    pub length: u32,
    pub teacher_confidence: f64,
}

#[derive(Debug, PartialEq)]
pub enum CurriculumVerdict {
    Ok { order: Vec<usize> },
    EmptySamples,
    InvalidValues,
}

pub fn order(samples: &[Sample]) -> CurriculumVerdict {
    if samples.is_empty() {
        return CurriculumVerdict::EmptySamples;
    }
    if samples.iter().any(|s| {
        !s.teacher_loss.is_finite()
            || !s.teacher_confidence.is_finite()
            || s.teacher_loss < 0.0
            || !(0.0..=1.0).contains(&s.teacher_confidence)
    }) {
        return CurriculumVerdict::InvalidValues;
    }
    let mut indexed: Vec<(usize, f64)> = samples
        .iter()
        .enumerate()
        .map(|(i, s)| {
            // Difficulty score: higher = harder.
            let length_norm = f64::from(s.length) / 1000.0;
            let confidence_inv = 1.0 - s.teacher_confidence;
            let score = s.teacher_loss + length_norm + confidence_inv;
            (i, score)
        })
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    CurriculumVerdict::Ok {
        order: indexed.into_iter().map(|(i, _)| i).collect(),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_curriculum_difficulty")?;

    let samples = [
        Sample {
            teacher_loss: 2.0,
            length: 200,
            teacher_confidence: 0.5,
        },
        Sample {
            teacher_loss: 0.5,
            length: 50,
            teacher_confidence: 0.95,
        },
        Sample {
            teacher_loss: 1.0,
            length: 100,
            teacher_confidence: 0.7,
        },
    ];
    println!("typical: {:?}", order(&samples));
    println!("empty: {:?}", order(&[]));
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
    fn easiest_first_in_order() {
        let samples = [
            Sample {
                teacher_loss: 2.0,
                length: 200,
                teacher_confidence: 0.5,
            },
            Sample {
                teacher_loss: 0.5,
                length: 50,
                teacher_confidence: 0.95,
            },
        ];
        if let CurriculumVerdict::Ok { order } = order(&samples) {
            // Index 1 (easier) should come first.
            assert_eq!(order, vec![1, 0]);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(order(&[]), CurriculumVerdict::EmptySamples);
    }

    #[test]
    fn invalid_negative_loss() {
        let bad = [Sample {
            teacher_loss: -1.0,
            length: 50,
            teacher_confidence: 0.5,
        }];
        assert_eq!(order(&bad), CurriculumVerdict::InvalidValues);
    }

    #[test]
    fn invalid_confidence_above_one() {
        let bad = [Sample {
            teacher_loss: 1.0,
            length: 50,
            teacher_confidence: 1.5,
        }];
        assert_eq!(order(&bad), CurriculumVerdict::InvalidValues);
    }

    #[test]
    fn nan_rejected() {
        let bad = [Sample {
            teacher_loss: f64::NAN,
            length: 50,
            teacher_confidence: 0.5,
        }];
        assert_eq!(order(&bad), CurriculumVerdict::InvalidValues);
    }

    #[test]
    fn order_length_matches_input() {
        let samples = [
            Sample {
                teacher_loss: 1.0,
                length: 50,
                teacher_confidence: 0.5,
            },
            Sample {
                teacher_loss: 2.0,
                length: 100,
                teacher_confidence: 0.4,
            },
            Sample {
                teacher_loss: 3.0,
                length: 200,
                teacher_confidence: 0.3,
            },
        ];
        if let CurriculumVerdict::Ok { order } = order(&samples) {
            assert_eq!(order.len(), 3);
        }
    }

    #[test]
    fn order_contains_all_indices() {
        let samples = [
            Sample {
                teacher_loss: 1.0,
                length: 50,
                teacher_confidence: 0.5,
            },
            Sample {
                teacher_loss: 2.0,
                length: 100,
                teacher_confidence: 0.4,
            },
        ];
        if let CurriculumVerdict::Ok { order } = order(&samples) {
            let set: std::collections::BTreeSet<_> = order.iter().collect();
            assert_eq!(set.len(), 2);
        }
    }

    #[test]
    fn high_confidence_easier() {
        // Two samples, identical except confidence.
        let samples = [
            Sample {
                teacher_loss: 1.0,
                length: 50,
                teacher_confidence: 0.5,
            },
            Sample {
                teacher_loss: 1.0,
                length: 50,
                teacher_confidence: 0.95,
            },
        ];
        if let CurriculumVerdict::Ok { order } = order(&samples) {
            // Higher confidence (index 1) should be easier → first.
            assert_eq!(order[0], 1);
        }
    }

    #[test]
    fn shorter_samples_easier() {
        // Two samples, identical except length.
        let samples = [
            Sample {
                teacher_loss: 1.0,
                length: 500,
                teacher_confidence: 0.5,
            },
            Sample {
                teacher_loss: 1.0,
                length: 50,
                teacher_confidence: 0.5,
            },
        ];
        if let CurriculumVerdict::Ok { order } = order(&samples) {
            assert_eq!(order[0], 1);
        }
    }

    #[test]
    fn lower_loss_easier() {
        let samples = [
            Sample {
                teacher_loss: 5.0,
                length: 50,
                teacher_confidence: 0.5,
            },
            Sample {
                teacher_loss: 0.5,
                length: 50,
                teacher_confidence: 0.5,
            },
        ];
        if let CurriculumVerdict::Ok { order } = order(&samples) {
            assert_eq!(order[0], 1);
        }
    }
}
