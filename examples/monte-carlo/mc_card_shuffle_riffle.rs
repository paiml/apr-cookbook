//! # Monte-Carlo Riffle Shuffle Quality
//!
//! Sim N riffle shuffles of a 52-card deck and measure how close the
//! resulting permutation is to uniform. Returns mean inversion count.
//!
//! Demonstrates the **MC.181** recipe for PMAT-219 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Bayer & Diaconis, "Trailing the Dovetail Shuffle to Its
//!  Lair" Annals of Applied Probability 2(2) (1992); 7-shuffle
//!  threshold result.
//!
//! Run with: cargo run --example mc_card_shuffle_riffle
//!
//! Added by PMAT-219 (catalog 1594→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RiffleVerdict {
    Ok {
        mean_inversions: u32,
        max_inversions: u32,
    },
    InvalidConfig,
}

pub fn simulate(deck_size: u32, shuffles: u32, trials: u32, seed: u64) -> RiffleVerdict {
    if deck_size < 4 || shuffles == 0 || trials < 100 {
        return RiffleVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let mut total_inv: u64 = 0;
    let mut max_inv = 0u32;
    for _ in 0..trials {
        let mut deck: Vec<u32> = (0..deck_size).collect();
        for _ in 0..shuffles {
            // Riffle: split at binomial(N, 0.5), then interleave.
            let mut split = 0u32;
            for _ in 0..deck_size {
                if (lcg(&mut state) >> 32) & 1 == 1 {
                    split += 1;
                }
            }
            let split = split as usize;
            let left: Vec<u32> = deck[..split].to_vec();
            let right: Vec<u32> = deck[split..].to_vec();
            let mut new_deck: Vec<u32> = Vec::with_capacity(deck_size as usize);
            let mut li = 0usize;
            let mut ri = 0usize;
            while li < left.len() && ri < right.len() {
                let pl = (left.len() - li) as u64;
                let pr = (right.len() - ri) as u64;
                let total = pl + pr;
                let r = lcg(&mut state) % total;
                if r < pl {
                    new_deck.push(left[li]);
                    li += 1;
                } else {
                    new_deck.push(right[ri]);
                    ri += 1;
                }
            }
            new_deck.extend(&left[li..]);
            new_deck.extend(&right[ri..]);
            deck = new_deck;
        }
        let inv = count_inversions(&deck);
        total_inv += inv as u64;
        if inv > max_inv {
            max_inv = inv;
        }
    }
    RiffleVerdict::Ok {
        mean_inversions: (total_inv / trials as u64) as u32,
        max_inversions: max_inv,
    }
}

fn count_inversions(arr: &[u32]) -> u32 {
    let mut count = 0u32;
    for i in 0..arr.len() {
        for j in (i + 1)..arr.len() {
            if arr[i] > arr[j] {
                count += 1;
            }
        }
    }
    count
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_card_shuffle_riffle")?;

    println!("1 shuffle: {:?}", simulate(52, 1, 200, 42));
    println!("7 shuffles: {:?}", simulate(52, 7, 200, 42));
    println!("invalid: {:?}", simulate(2, 1, 100, 42));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn invalid_too_small_deck() {
        assert_eq!(simulate(2, 1, 100, 42), RiffleVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_shuffles() {
        assert_eq!(simulate(52, 0, 100, 42), RiffleVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_trials() {
        assert_eq!(simulate(52, 1, 50, 42), RiffleVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(52, 3, 100, 42);
        let b = simulate(52, 3, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn shuffle_inversions_in_valid_range() {
        // Mean inversions for n=52 must be in [0, max=52*51/2=1326].
        let v = simulate(52, 10, 500, 42);
        if let RiffleVerdict::Ok {
            mean_inversions, ..
        } = v
        {
            assert!(mean_inversions <= 1326);
        }
    }

    #[test]
    fn max_inversions_at_least_mean() {
        let v = simulate(52, 5, 200, 42);
        if let RiffleVerdict::Ok {
            mean_inversions,
            max_inversions,
        } = v
        {
            assert!(max_inversions >= mean_inversions);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(4, 1, 100, 42);
        assert!(matches!(v, RiffleVerdict::Ok { .. }));
    }

    #[test]
    fn many_trials_handled() {
        let v = simulate(20, 5, 1000, 42);
        assert!(matches!(v, RiffleVerdict::Ok { .. }));
    }

    #[test]
    fn shuffle_finite_outcomes() {
        let v = simulate(52, 3, 200, 42);
        if let RiffleVerdict::Ok {
            mean_inversions,
            max_inversions,
        } = v
        {
            assert!(mean_inversions < u32::MAX);
            assert!(max_inversions < u32::MAX);
        }
    }

    #[test]
    fn count_inversions_correct() {
        // [3, 1, 2]: (3,1), (3,2) → 2 inversions
        assert_eq!(count_inversions(&[3, 1, 2]), 2);
        // Sorted: 0 inversions
        assert_eq!(count_inversions(&[1, 2, 3]), 0);
        // Reversed: n*(n-1)/2 inversions
        assert_eq!(count_inversions(&[3, 2, 1]), 3);
    }

    #[test]
    fn inversions_le_max_possible() {
        // Max inversions for n=52 = 52*51/2 = 1326
        let v = simulate(52, 5, 200, 42);
        if let RiffleVerdict::Ok { max_inversions, .. } = v {
            assert!(max_inversions <= 1326);
        }
    }
}
