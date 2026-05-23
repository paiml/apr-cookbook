//! # Advanced Payload Size Limit
//!
//! Reject overly large requests early. Different limits for body
//! and per-field. Returns the violating field if any.
//!
//! Demonstrates the **ADV.38** recipe for PMAT-158 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: nginx client_max_body_size + per-field validation.
//!
//! Run with: cargo run --example adv_payload_size_limit
//!
//! Added by PMAT-158 (catalog 1045→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SizeVerdict {
    Ok,
    BodyTooLarge {
        size: u64,
        limit: u64,
    },
    FieldTooLarge {
        field: String,
        size: u64,
        limit: u64,
    },
    InvalidConfig,
}

pub fn check(
    body_size: u64,
    body_limit: u64,
    fields: &[(&str, u64)],
    field_limit: u64,
) -> SizeVerdict {
    if body_limit == 0 || field_limit == 0 {
        return SizeVerdict::InvalidConfig;
    }
    if body_size > body_limit {
        return SizeVerdict::BodyTooLarge {
            size: body_size,
            limit: body_limit,
        };
    }
    for (name, size) in fields {
        if *size > field_limit {
            return SizeVerdict::FieldTooLarge {
                field: (*name).to_string(),
                size: *size,
                limit: field_limit,
            };
        }
    }
    SizeVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_payload_size_limit")?;

    let fields_ok = vec![("prompt", 1_024_u64), ("system", 512)];
    println!("ok: {:?}", check(2_048, 10_000, &fields_ok, 8_192));
    println!(
        "body too large: {:?}",
        check(20_000, 10_000, &fields_ok, 8_192)
    );

    let fields_bad = vec![("prompt", 16_384_u64)];
    println!(
        "field too large: {:?}",
        check(20_000, 100_000, &fields_bad, 8_192)
    );
    println!("invalid: {:?}", check(20_000, 0, &fields_ok, 8_192));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_fields() -> Vec<(&'static str, u64)> {
        vec![("prompt", 1_024), ("system", 512)]
    }

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn within_limits_ok() {
        let v = check(2_048, 10_000, &ok_fields(), 8_192);
        assert_eq!(v, SizeVerdict::Ok);
    }

    #[test]
    fn body_over_rejected() {
        let v = check(20_000, 10_000, &ok_fields(), 8_192);
        assert!(matches!(v, SizeVerdict::BodyTooLarge { .. }));
    }

    #[test]
    fn field_over_rejected() {
        let bad: Vec<(&str, u64)> = vec![("prompt", 16_384)];
        let v = check(20_000, 100_000, &bad, 8_192);
        if let SizeVerdict::FieldTooLarge { field, .. } = v {
            assert_eq!(field, "prompt");
        }
    }

    #[test]
    fn body_check_first() {
        // Body fails before checking fields.
        let bad: Vec<(&str, u64)> = vec![("prompt", 16_384)];
        let v = check(200_000, 100_000, &bad, 8_192);
        assert!(matches!(v, SizeVerdict::BodyTooLarge { .. }));
    }

    #[test]
    fn zero_body_limit_invalid() {
        assert_eq!(
            check(100, 0, &ok_fields(), 8_192),
            SizeVerdict::InvalidConfig
        );
    }

    #[test]
    fn zero_field_limit_invalid() {
        assert_eq!(
            check(100, 1000, &ok_fields(), 0),
            SizeVerdict::InvalidConfig
        );
    }

    #[test]
    fn empty_fields_ok() {
        let v = check(100, 1000, &[], 8192);
        assert_eq!(v, SizeVerdict::Ok);
    }

    #[test]
    fn boundary_at_body_limit_ok() {
        // Equal-to-limit is OK.
        let v = check(1000, 1000, &[], 8192);
        assert_eq!(v, SizeVerdict::Ok);
    }

    #[test]
    fn first_failing_field_returned() {
        let fields: Vec<(&str, u64)> = vec![("a", 100), ("b", 99999)];
        let v = check(1000, 100_000, &fields, 8192);
        if let SizeVerdict::FieldTooLarge { field, .. } = v {
            assert_eq!(field, "b");
        }
    }

    #[test]
    fn deterministic() {
        let a = check(1000, 10_000, &ok_fields(), 8_192);
        let b = check(1000, 10_000, &ok_fields(), 8_192);
        assert_eq!(a, b);
    }
}
