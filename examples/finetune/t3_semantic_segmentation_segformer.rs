//! # Tier 3.13 — SegFormer semantic segmentation (tabular-only)
//!
//! Falsifier: SegFormer per-pixel predictions on a fixture yield mIoU ≥ 0.5
//! when the model predicts >50% of pixels correctly.
//!
//! Run with: cargo run --example t3_semantic_segmentation_segformer

use apr_cookbook::finetune::specialty;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const N_CLASSES: u8 = 4;

fn fixture() -> (Vec<u8>, Vec<u8>) {
    // 80% correct: alternating 4-class pattern with 20% errors.
    let mut p = Vec::new();
    let mut t = Vec::new();
    for i in 0..100 {
        let class = (i % 4) as u8;
        t.push(class);
        let predicted = if i % 5 == 0 { (class + 1) % 4 } else { class };
        p.push(predicted);
    }
    (p, t)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_semantic_segmentation_segformer")?;
    let (p, t) = fixture();
    let miou = specialty::segformer_miou(&p, &t, N_CLASSES);
    println!("✓ SegFormer mIoU = {:.4}", miou);
    assert!(miou >= 0.5);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn falsifier_holds_on_fixture() {
        let (p, t) = fixture();
        assert!(specialty::segformer_miou(&p, &t, N_CLASSES) >= 0.5);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // All predictions wrong → mIoU = 0.
        let p = vec![0_u8; 100];
        let t = vec![1_u8; 100];
        assert_eq!(specialty::segformer_miou(&p, &t, 2), 0.0);
    }

    #[test]
    fn deterministic_across_runs() {
        let (p, t) = fixture();
        let a = specialty::segformer_miou(&p, &t, N_CLASSES);
        let b = specialty::segformer_miou(&p, &t, N_CLASSES);
        assert_eq!(a, b);
    }
}
