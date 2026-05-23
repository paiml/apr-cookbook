//! # Contracts-Macros Purity Check
//!
//! Static checker for IIUR recipe purity claims. Given a list of
//! observed effects (env_read, network, file_io, panic, etc.), verify
//! none are forbidden by IIUR contract.
//!
//! Demonstrates the **CMM.15** recipe for PMAT-162 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Effect systems (Koka, Eff lang) + IIUR purity invariants.
//!
//! Run with: cargo run --example contracts_macros_pure_check
//!
//! Added by PMAT-162 (catalog 1081→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
    EnvRead,
    Network,
    FileWrite,
    FileRead,
    Panic,
    Random,
    Time,
    Allocation,
}

#[derive(Debug, PartialEq)]
pub enum PurityVerdict {
    Pure,
    Impure { violating: Vec<Effect> },
}

pub fn check(observed: &[Effect]) -> PurityVerdict {
    // IIUR forbids env, network, file write, panic, time, but allows
    // allocations (deterministic Vec::new) and one-shot randomness via seed.
    let forbidden: &[Effect] = &[
        Effect::EnvRead,
        Effect::Network,
        Effect::FileWrite,
        Effect::Panic,
        Effect::Time,
    ];
    let violating: Vec<Effect> = observed
        .iter()
        .copied()
        .filter(|e| forbidden.contains(e))
        .collect();
    if violating.is_empty() {
        PurityVerdict::Pure
    } else {
        PurityVerdict::Impure { violating }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_pure_check")?;

    println!("pure: {:?}", check(&[Effect::Allocation, Effect::Random]));
    println!("impure: {:?}", check(&[Effect::EnvRead, Effect::Network]));
    println!("file read ok: {:?}", check(&[Effect::FileRead]));
    println!("empty: {:?}", check(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn no_effects_pure() {
        assert_eq!(check(&[]), PurityVerdict::Pure);
    }

    #[test]
    fn allocation_pure() {
        assert_eq!(check(&[Effect::Allocation]), PurityVerdict::Pure);
    }

    #[test]
    fn random_pure() {
        // Random is allowed (deterministic via seed).
        assert_eq!(check(&[Effect::Random]), PurityVerdict::Pure);
    }

    #[test]
    fn file_read_pure() {
        // FileRead allowed (loading model, etc.).
        assert_eq!(check(&[Effect::FileRead]), PurityVerdict::Pure);
    }

    #[test]
    fn env_read_impure() {
        let v = check(&[Effect::EnvRead]);
        if let PurityVerdict::Impure { violating } = v {
            assert_eq!(violating, vec![Effect::EnvRead]);
        }
    }

    #[test]
    fn network_impure() {
        assert!(matches!(
            check(&[Effect::Network]),
            PurityVerdict::Impure { .. }
        ));
    }

    #[test]
    fn file_write_impure() {
        assert!(matches!(
            check(&[Effect::FileWrite]),
            PurityVerdict::Impure { .. }
        ));
    }

    #[test]
    fn panic_impure() {
        assert!(matches!(
            check(&[Effect::Panic]),
            PurityVerdict::Impure { .. }
        ));
    }

    #[test]
    fn time_impure() {
        assert!(matches!(
            check(&[Effect::Time]),
            PurityVerdict::Impure { .. }
        ));
    }

    #[test]
    fn multiple_violations_listed() {
        let v = check(&[Effect::EnvRead, Effect::Network, Effect::FileWrite]);
        if let PurityVerdict::Impure { violating } = v {
            assert_eq!(violating.len(), 3);
        }
    }

    #[test]
    fn allowed_mixed_with_impure_still_impure() {
        let v = check(&[Effect::Allocation, Effect::EnvRead]);
        assert!(matches!(v, PurityVerdict::Impure { .. }));
    }

    #[test]
    fn deterministic() {
        let a = check(&[Effect::Network]);
        let b = check(&[Effect::Network]);
        assert_eq!(a, b);
    }
}
