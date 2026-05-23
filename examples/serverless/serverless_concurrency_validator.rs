//! # Serverless Concurrency Limit Validator
//!
//! AWS Lambda has account-level concurrent execution limit (1000
//! default, raisable). Reserved concurrency: per-function reservation;
//! Provisioned: pre-warmed instances. Constraints: reserved ≤ account
//! limit; provisioned ≤ reserved; sum of all reserved ≤ account.
//! This recipe builds the validator.
//!
//! Demonstrates the **SVL.7** recipe for PMAT-126 (serverless coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: AWS Lambda Concurrency Limits docs.
//!
//! Run with: cargo run --example serverless_concurrency_validator
//!
//! Added by PMAT-126 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ConcurrencyVerdict {
    Ok { headroom: u32 },
    ReservedExceedsAccount,
    ProvisionedExceedsReserved,
    SumReservedExceedsAccount { sum: u32, account: u32 },
}

pub fn validate_function(
    account_limit: u32,
    reserved: u32,
    provisioned: u32,
) -> ConcurrencyVerdict {
    if reserved > account_limit {
        return ConcurrencyVerdict::ReservedExceedsAccount;
    }
    if provisioned > reserved {
        return ConcurrencyVerdict::ProvisionedExceedsReserved;
    }
    ConcurrencyVerdict::Ok {
        headroom: account_limit - reserved,
    }
}

pub fn validate_account(account_limit: u32, per_function_reserved: &[u32]) -> ConcurrencyVerdict {
    let sum: u64 = per_function_reserved.iter().map(|&n| u64::from(n)).sum();
    if sum > u64::from(account_limit) {
        return ConcurrencyVerdict::SumReservedExceedsAccount {
            sum: sum.min(u64::from(u32::MAX)) as u32,
            account: account_limit,
        };
    }
    ConcurrencyVerdict::Ok {
        headroom: account_limit - sum as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("serverless_concurrency_validator")?;

    println!("ok: {:?}", validate_function(1000, 100, 50));
    println!("reserved exceeds: {:?}", validate_function(1000, 1500, 100));
    println!(
        "provisioned exceeds: {:?}",
        validate_function(1000, 100, 200)
    );
    println!(
        "account sum exceeds: {:?}",
        validate_account(1000, &[400, 400, 400])
    );
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
    fn typical_function_passes() {
        assert_eq!(
            validate_function(1000, 100, 50),
            ConcurrencyVerdict::Ok { headroom: 900 }
        );
    }

    #[test]
    fn reserved_exceeds_account_rejected() {
        assert_eq!(
            validate_function(1000, 1500, 100),
            ConcurrencyVerdict::ReservedExceedsAccount
        );
    }

    #[test]
    fn provisioned_exceeds_reserved_rejected() {
        assert_eq!(
            validate_function(1000, 100, 200),
            ConcurrencyVerdict::ProvisionedExceedsReserved
        );
    }

    #[test]
    fn provisioned_equal_to_reserved_passes() {
        // 100% pre-warmed is valid.
        assert_eq!(
            validate_function(1000, 100, 100),
            ConcurrencyVerdict::Ok { headroom: 900 }
        );
    }

    #[test]
    fn zero_reserved_zero_provisioned_passes() {
        // No reservation = burst capacity.
        assert_eq!(
            validate_function(1000, 0, 0),
            ConcurrencyVerdict::Ok { headroom: 1000 }
        );
    }

    #[test]
    fn account_sum_within_limit_passes() {
        assert_eq!(
            validate_account(1000, &[100, 200, 300]),
            ConcurrencyVerdict::Ok { headroom: 400 }
        );
    }

    #[test]
    fn account_sum_exceeds_rejected() {
        let v = validate_account(1000, &[400, 400, 400]);
        assert!(matches!(
            v,
            ConcurrencyVerdict::SumReservedExceedsAccount { .. }
        ));
    }

    #[test]
    fn empty_per_function_uses_full_headroom() {
        assert_eq!(
            validate_account(1000, &[]),
            ConcurrencyVerdict::Ok { headroom: 1000 }
        );
    }

    #[test]
    fn reserved_at_account_limit_passes() {
        assert_eq!(
            validate_function(1000, 1000, 500),
            ConcurrencyVerdict::Ok { headroom: 0 }
        );
    }
}
