//! # Distillation Data-Augmentation Picker
//!
//! Augmentation strategies for KD set:
//!   None: clean teacher labels (best for tiny datasets)
//!   Mixup: convex combination of pairs (helps generalization)
//!   CutMix: spatial mixup (vision)
//!   RandAugment: random ops chain (heavy aug for big datasets)
//!
//! Picker rules:
//!   dataset < 1k → None
//!   1k-100k + classification → Mixup
//!   1k-100k + vision-spatial → CutMix
//!   ≥ 100k → RandAugment
//!
//! Demonstrates the **DIST.17** recipe for PMAT-145 (distillation round 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Cubuk et al. (2020). RandAugment: Practical Automated Data Augmentation.
//!
//! Run with: cargo run --example distill_data_augmentation
//!
//! Added by PMAT-145 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskKind {
    TextClassification,
    ImageClassification,
    Detection,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AugStrategy {
    None,
    Mixup,
    CutMix,
    RandAugment,
}

#[derive(Debug, PartialEq)]
pub enum AugVerdict {
    Ok { strategy: AugStrategy, alpha: f64 },
    InvalidDatasetSize,
}

pub fn pick(dataset_size: u32, task: TaskKind) -> AugVerdict {
    if dataset_size == 0 {
        return AugVerdict::InvalidDatasetSize;
    }
    let strategy = if dataset_size < 1_000 {
        AugStrategy::None
    } else if dataset_size >= 100_000 {
        AugStrategy::RandAugment
    } else if matches!(task, TaskKind::ImageClassification | TaskKind::Detection) {
        AugStrategy::CutMix
    } else {
        AugStrategy::Mixup
    };
    let alpha = match strategy {
        AugStrategy::None => 0.0,
        AugStrategy::Mixup | AugStrategy::CutMix => 0.2,
        AugStrategy::RandAugment => 0.5,
    };
    AugVerdict::Ok { strategy, alpha }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_data_augmentation")?;

    println!("tiny set: {:?}", pick(500, TaskKind::TextClassification));
    println!(
        "medium text: {:?}",
        pick(10_000, TaskKind::TextClassification)
    );
    println!(
        "medium image: {:?}",
        pick(10_000, TaskKind::ImageClassification)
    );
    println!(
        "large set: {:?}",
        pick(500_000, TaskKind::ImageClassification)
    );
    println!("zero: {:?}", pick(0, TaskKind::Other));
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
    fn tiny_dataset_no_aug() {
        let v = pick(500, TaskKind::TextClassification);
        if let AugVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, AugStrategy::None);
        }
    }

    #[test]
    fn medium_text_uses_mixup() {
        let v = pick(10_000, TaskKind::TextClassification);
        if let AugVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, AugStrategy::Mixup);
        }
    }

    #[test]
    fn medium_image_uses_cutmix() {
        let v = pick(10_000, TaskKind::ImageClassification);
        if let AugVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, AugStrategy::CutMix);
        }
    }

    #[test]
    fn large_uses_rand_augment() {
        let v = pick(500_000, TaskKind::ImageClassification);
        if let AugVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, AugStrategy::RandAugment);
        }
    }

    #[test]
    fn detection_treated_as_image() {
        let v = pick(10_000, TaskKind::Detection);
        if let AugVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, AugStrategy::CutMix);
        }
    }

    #[test]
    fn other_task_uses_mixup_in_medium() {
        let v = pick(10_000, TaskKind::Other);
        if let AugVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, AugStrategy::Mixup);
        }
    }

    #[test]
    fn zero_size_invalid() {
        assert_eq!(
            pick(0, TaskKind::TextClassification),
            AugVerdict::InvalidDatasetSize
        );
    }

    #[test]
    fn alpha_zero_for_none() {
        if let AugVerdict::Ok { alpha, .. } = pick(500, TaskKind::TextClassification) {
            assert_eq!(alpha, 0.0);
        }
    }

    #[test]
    fn alpha_higher_for_rand_augment() {
        let v_mid = pick(10_000, TaskKind::TextClassification);
        let v_large = pick(500_000, TaskKind::ImageClassification);
        if let (AugVerdict::Ok { alpha: m, .. }, AugVerdict::Ok { alpha: l, .. }) = (v_mid, v_large)
        {
            assert!(l > m);
        }
    }

    #[test]
    fn boundary_at_1k_uses_aug() {
        // exactly 1000 → no longer "tiny", so aug applies.
        let v = pick(1_000, TaskKind::TextClassification);
        if let AugVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, AugStrategy::Mixup);
        }
    }
}
