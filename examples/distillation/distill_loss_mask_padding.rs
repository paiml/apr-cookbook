//! # Distillation Padding-Token Loss Mask
//!
//! In sequence distillation, padding tokens (PAD) should not contribute
//! to the loss. Build a mask that zeroes loss at PAD positions, then
//! return mean loss over real tokens only.
//!
//! Demonstrates the **DIST.34** recipe for PMAT-157 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Hugging Face Trainer label-padding mask convention.
//!
//! Run with: cargo run --example distill_loss_mask_padding
//!
//! Added by PMAT-157 (catalog 1036→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum MaskVerdict {
    Ok {
        masked_mean_loss: f64,
        real_token_count: u32,
    },
    AllPadding,
    LengthMismatch,
    InvalidLoss,
}

pub fn apply(per_token_loss: &[f64], token_ids: &[u32], pad_id: u32) -> MaskVerdict {
    if per_token_loss.len() != token_ids.len() {
        return MaskVerdict::LengthMismatch;
    }
    if per_token_loss.iter().any(|l| !l.is_finite() || *l < 0.0) {
        return MaskVerdict::InvalidLoss;
    }
    let mut sum = 0.0;
    let mut count = 0u32;
    for (loss, &tok) in per_token_loss.iter().zip(token_ids.iter()) {
        if tok != pad_id {
            sum += loss;
            count += 1;
        }
    }
    if count == 0 {
        return MaskVerdict::AllPadding;
    }
    MaskVerdict::Ok {
        masked_mean_loss: sum / f64::from(count),
        real_token_count: count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_loss_mask_padding")?;

    let losses = vec![0.5, 0.3, 0.4, 0.0, 0.0];
    let tokens = vec![10, 20, 30, 0, 0];
    println!("typical: {:?}", apply(&losses, &tokens, 0));
    println!("all pad: {:?}", apply(&[0.5, 0.3], &[0, 0], 0));
    println!("mismatch: {:?}", apply(&[0.5], &[0, 0], 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn padding_excluded_from_mean() {
        let v = apply(&[1.0, 1.0, 1.0, 100.0], &[10, 20, 30, 0], 0);
        if let MaskVerdict::Ok {
            masked_mean_loss,
            real_token_count,
        } = v
        {
            // Padding contributes 100.0 but is masked.
            assert!((masked_mean_loss - 1.0).abs() < 1e-9);
            assert_eq!(real_token_count, 3);
        }
    }

    #[test]
    fn no_padding_uses_all_tokens() {
        let v = apply(&[1.0, 2.0, 3.0], &[10, 20, 30], 0);
        if let MaskVerdict::Ok {
            real_token_count, ..
        } = v
        {
            assert_eq!(real_token_count, 3);
        }
    }

    #[test]
    fn all_padding_rejected() {
        assert_eq!(apply(&[0.5, 0.3], &[0, 0], 0), MaskVerdict::AllPadding);
    }

    #[test]
    fn length_mismatch_rejected() {
        assert_eq!(apply(&[0.5], &[0, 0], 0), MaskVerdict::LengthMismatch);
    }

    #[test]
    fn empty_inputs_all_padding() {
        // Empty zip-no-real-tokens path.
        assert_eq!(apply(&[], &[], 0), MaskVerdict::AllPadding);
    }

    #[test]
    fn nan_loss_rejected() {
        assert_eq!(
            apply(&[f64::NAN, 0.0], &[10, 20], 0),
            MaskVerdict::InvalidLoss
        );
    }

    #[test]
    fn negative_loss_rejected() {
        assert_eq!(apply(&[-1.0, 1.0], &[10, 20], 0), MaskVerdict::InvalidLoss);
    }

    #[test]
    fn pad_id_other_than_zero() {
        let v = apply(&[1.0, 1.0, 1.0], &[10, 99, 20], 99);
        if let MaskVerdict::Ok {
            masked_mean_loss,
            real_token_count,
        } = v
        {
            assert_eq!(real_token_count, 2);
            assert!((masked_mean_loss - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn mean_correct_for_mixed() {
        let v = apply(&[2.0, 4.0, 6.0], &[10, 20, 30], 0);
        if let MaskVerdict::Ok {
            masked_mean_loss, ..
        } = v
        {
            assert!((masked_mean_loss - 4.0).abs() < 1e-9);
        }
    }

    #[test]
    fn deterministic() {
        let losses = vec![1.0, 1.0, 1.0];
        let tokens = vec![10, 20, 30];
        let a = apply(&losses, &tokens, 0);
        let b = apply(&losses, &tokens, 0);
        assert_eq!(a, b);
    }
}
