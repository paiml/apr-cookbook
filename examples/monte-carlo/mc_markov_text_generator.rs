//! # Monte-Carlo Markov Chain Text Generator
//!
//! Build a 1st-order character Markov chain from training text and
//! generate samples. Returns the generated string and observed
//! transition count.
//!
//! Demonstrates the **MC.157** recipe for PMAT-211 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Shannon, "A Mathematical Theory of Communication" §3
//!  (1948); Markov chain language models — predecessor to n-grams.
//!
//! Run with: cargo run --example mc_markov_text_generator
//!
//! Added by PMAT-211 (catalog 1522→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum MarkovVerdict {
    Ok {
        generated: String,
        transitions_used: u32,
    },
    InvalidConfig,
}

pub fn generate(training: &str, output_len: u32, seed: u64) -> MarkovVerdict {
    if training.len() < 2 || output_len == 0 {
        return MarkovVerdict::InvalidConfig;
    }
    // Build transition table: char → list of next chars.
    let mut table: BTreeMap<char, Vec<char>> = BTreeMap::new();
    let chars: Vec<char> = training.chars().collect();
    for w in chars.windows(2) {
        table.entry(w[0]).or_default().push(w[1]);
    }
    let mut state = seed | 1;
    let start = chars[0];
    let mut output = String::with_capacity(output_len as usize);
    output.push(start);
    let mut current = start;
    let mut transitions = 0u32;
    for _ in 1..output_len {
        let next = match table.get(&current) {
            Some(options) if !options.is_empty() => {
                let idx = (lcg(&mut state) as usize) % options.len();
                options[idx]
            }
            _ => start, // dead-end → restart
        };
        output.push(next);
        current = next;
        transitions += 1;
    }
    MarkovVerdict::Ok {
        generated: output,
        transitions_used: transitions,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_markov_text_generator")?;

    println!("text: {:?}", generate("the quick brown fox jumps", 30, 42));
    println!("invalid: {:?}", generate("a", 5, 42));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn invalid_short_training() {
        assert_eq!(generate("a", 5, 42), MarkovVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_output() {
        assert_eq!(generate("ab", 0, 42), MarkovVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = generate("the cat sat", 20, 42);
        let b = generate("the cat sat", 20, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn output_length_matches() {
        let v = generate("the cat sat", 10, 42);
        if let MarkovVerdict::Ok { generated, .. } = v {
            assert_eq!(generated.chars().count(), 10);
        }
    }

    #[test]
    fn first_char_matches_training() {
        let v = generate("zebra", 5, 42);
        if let MarkovVerdict::Ok { generated, .. } = v {
            assert_eq!(generated.chars().next(), Some('z'));
        }
    }

    #[test]
    fn transitions_count_correct() {
        let v = generate("ab", 5, 42);
        if let MarkovVerdict::Ok {
            transitions_used, ..
        } = v
        {
            assert_eq!(transitions_used, 4);
        }
    }

    #[test]
    fn cyclic_training_handled() {
        // "ababab" → strict alternation
        let v = generate("ababab", 6, 42);
        if let MarkovVerdict::Ok { generated, .. } = v {
            // Expect alternating a/b pattern.
            assert!(generated.contains('a'));
            assert!(generated.contains('b'));
        }
    }

    #[test]
    fn long_training_handled() {
        let training = "the quick brown fox jumps over the lazy dog and runs";
        let v = generate(training, 50, 42);
        if let MarkovVerdict::Ok { generated, .. } = v {
            assert_eq!(generated.chars().count(), 50);
        }
    }

    #[test]
    fn different_seeds_different_output() {
        let a = generate("the quick brown fox", 30, 42);
        let b = generate("the quick brown fox", 30, 999);
        assert!(a != b);
    }

    #[test]
    fn min_inputs_accepted() {
        let v = generate("ab", 1, 42);
        if let MarkovVerdict::Ok { generated, .. } = v {
            assert_eq!(generated.chars().count(), 1);
        }
    }

    #[test]
    fn unicode_training_handled() {
        let v = generate("caféau", 10, 42);
        if let MarkovVerdict::Ok { generated, .. } = v {
            assert_eq!(generated.chars().count(), 10);
        }
    }
}
