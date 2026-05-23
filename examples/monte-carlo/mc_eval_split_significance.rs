//! # Monte-Carlo Eval Split Significance
//!
//! Permutation test for "is model A significantly better than B?" on
//! shared eval set. Returns the empirical p-value and whether the
//! observed accuracy diff is significant at the chosen alpha.
//!
//! Demonstrates the **MC.23** recipe for PMAT-165 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Permutation-test for paired observations (Edgington 1969).
//!
//! Run with: cargo run --example mc_eval_split_significance
//!
//! Added by PMAT-165 (catalog 1108→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SignificanceVerdict {
    Significant { p_value: f64 },
    NotSignificant { p_value: f64 },
    InvalidConfig,
}

pub fn test(
    model_a_correct: &[bool],
    model_b_correct: &[bool],
    alpha: f64,
    permutations: u32,
    seed: u64,
) -> SignificanceVerdict {
    if model_a_correct.is_empty()
        || model_a_correct.len() != model_b_correct.len()
        || permutations == 0
        || !alpha.is_finite()
        || !(0.0..=1.0).contains(&alpha)
    {
        return SignificanceVerdict::InvalidConfig;
    }
    let n = model_a_correct.len();
    let observed_diff = (0..n)
        .map(|i| i64::from(model_a_correct[i]) - i64::from(model_b_correct[i]))
        .sum::<i64>()
        .abs();
    let mut rng_state = seed | 1;
    let mut as_or_more_extreme: u32 = 0;
    for _ in 0..permutations {
        let mut diff: i64 = 0;
        for i in 0..n {
            let swap = unit(&mut rng_state) < 0.5;
            let (a, b) = if swap {
                (model_b_correct[i], model_a_correct[i])
            } else {
                (model_a_correct[i], model_b_correct[i])
            };
            diff += i64::from(a) - i64::from(b);
        }
        if diff.abs() >= observed_diff {
            as_or_more_extreme += 1;
        }
    }
    let p_value = f64::from(as_or_more_extreme) / f64::from(permutations);
    if p_value < alpha {
        SignificanceVerdict::Significant { p_value }
    } else {
        SignificanceVerdict::NotSignificant { p_value }
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_eval_split_significance")?;

    let a = vec![true; 90]
        .into_iter()
        .chain(vec![false; 10])
        .collect::<Vec<_>>();
    let b = vec![true; 50]
        .into_iter()
        .chain(vec![false; 50])
        .collect::<Vec<_>>();
    println!("clearly different: {:?}", test(&a, &b, 0.05, 1000, 42));

    let same = vec![true; 60]
        .into_iter()
        .chain(vec![false; 40])
        .collect::<Vec<_>>();
    let same2 = same.clone();
    println!("same: {:?}", test(&same, &same2, 0.05, 1000, 42));

    println!(
        "invalid len: {:?}",
        test(&[true], &[true, false], 0.05, 100, 42)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tester_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn clear_difference_significant() {
        let a: Vec<bool> = vec![true; 90].into_iter().chain(vec![false; 10]).collect();
        let b: Vec<bool> = vec![true; 50].into_iter().chain(vec![false; 50]).collect();
        let v = test(&a, &b, 0.05, 1000, 42);
        assert!(matches!(v, SignificanceVerdict::Significant { .. }));
    }

    #[test]
    fn identical_not_significant() {
        let a = vec![true; 50]
            .into_iter()
            .chain(vec![false; 50])
            .collect::<Vec<_>>();
        let b = a.clone();
        let v = test(&a, &b, 0.05, 1000, 42);
        assert!(matches!(v, SignificanceVerdict::NotSignificant { .. }));
    }

    #[test]
    fn empty_invalid() {
        assert_eq!(
            test(&[], &[], 0.05, 100, 42),
            SignificanceVerdict::InvalidConfig
        );
    }

    #[test]
    fn length_mismatch_invalid() {
        assert_eq!(
            test(&[true], &[true, false], 0.05, 100, 42),
            SignificanceVerdict::InvalidConfig
        );
    }

    #[test]
    fn zero_permutations_invalid() {
        assert_eq!(
            test(&[true], &[false], 0.05, 0, 42),
            SignificanceVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_alpha() {
        assert_eq!(
            test(&[true], &[false], 1.5, 100, 42),
            SignificanceVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_alpha() {
        assert_eq!(
            test(&[true], &[false], f64::NAN, 100, 42),
            SignificanceVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = vec![true, false, true, false];
        let b = vec![false, false, true, true];
        let v1 = test(&a, &b, 0.05, 100, 42);
        let v2 = test(&a, &b, 0.05, 100, 42);
        assert_eq!(v1, v2);
    }

    #[test]
    fn p_value_in_unit_range() {
        let a = vec![true, true, false];
        let b = vec![false, true, false];
        let v = test(&a, &b, 0.05, 100, 42);
        if let SignificanceVerdict::Significant { p_value }
        | SignificanceVerdict::NotSignificant { p_value } = v
        {
            assert!((0.0..=1.0).contains(&p_value));
        }
    }

    #[test]
    fn one_pair_works() {
        let v = test(&[true], &[false], 0.05, 100, 42);
        assert!(matches!(
            v,
            SignificanceVerdict::Significant { .. } | SignificanceVerdict::NotSignificant { .. }
        ));
    }
}
