//! # Contracts-Macros Recipe Revision Bump
//!
//! Validate semver-style revision bump from `prev` to `next`. Allowed
//! bumps: patch (z+1), minor (y+1, z=0), major (x+1, y=0, z=0).
//! Returns bump kind or `Invalid` verdict.
//!
//! Demonstrates the **CMM.140** recipe for PMAT-204 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SemVer 2.0 spec; conventional-commits BREAKING CHANGE
//!  rules.
//!
//! Run with: cargo run --example contracts_macros_recipe_revision_bump
//!
//! Added by PMAT-204 (catalog 1459→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BumpVerdict {
    Patch,
    Minor,
    Major,
    NoChange,
    Invalid,
}

pub fn classify(prev: (u32, u32, u32), next: (u32, u32, u32)) -> BumpVerdict {
    if prev == next {
        return BumpVerdict::NoChange;
    }
    let (px, py, pz) = prev;
    let (nx, ny, nz) = next;
    if nx == px && ny == py && nz == pz + 1 {
        BumpVerdict::Patch
    } else if nx == px && ny == py + 1 && nz == 0 {
        BumpVerdict::Minor
    } else if nx == px + 1 && ny == 0 && nz == 0 {
        BumpVerdict::Major
    } else {
        BumpVerdict::Invalid
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_revision_bump")?;

    println!("patch: {:?}", classify((1, 2, 3), (1, 2, 4)));
    println!("minor: {:?}", classify((1, 2, 3), (1, 3, 0)));
    println!("major: {:?}", classify((1, 2, 3), (2, 0, 0)));
    println!("invalid: {:?}", classify((1, 2, 3), (3, 0, 0)));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn patch_bump() {
        assert_eq!(classify((1, 2, 3), (1, 2, 4)), BumpVerdict::Patch);
    }

    #[test]
    fn minor_bump_resets_patch() {
        assert_eq!(classify((1, 2, 3), (1, 3, 0)), BumpVerdict::Minor);
    }

    #[test]
    fn major_bump_resets_minor_patch() {
        assert_eq!(classify((1, 2, 3), (2, 0, 0)), BumpVerdict::Major);
    }

    #[test]
    fn no_change_handled() {
        assert_eq!(classify((1, 2, 3), (1, 2, 3)), BumpVerdict::NoChange);
    }

    #[test]
    fn skip_patch_invalid() {
        assert_eq!(classify((1, 2, 3), (1, 2, 5)), BumpVerdict::Invalid);
    }

    #[test]
    fn skip_minor_invalid() {
        assert_eq!(classify((1, 2, 3), (1, 4, 0)), BumpVerdict::Invalid);
    }

    #[test]
    fn skip_major_invalid() {
        assert_eq!(classify((1, 2, 3), (3, 0, 0)), BumpVerdict::Invalid);
    }

    #[test]
    fn minor_without_zero_patch_invalid() {
        assert_eq!(classify((1, 2, 3), (1, 3, 5)), BumpVerdict::Invalid);
    }

    #[test]
    fn major_without_zero_minor_invalid() {
        assert_eq!(classify((1, 2, 3), (2, 1, 0)), BumpVerdict::Invalid);
    }

    #[test]
    fn deterministic() {
        let r1 = classify((1, 2, 3), (1, 2, 4));
        let r2 = classify((1, 2, 3), (1, 2, 4));
        assert_eq!(r1, r2);
    }

    #[test]
    fn downgrade_invalid() {
        assert_eq!(classify((1, 2, 3), (1, 2, 2)), BumpVerdict::Invalid);
    }

    #[test]
    fn from_zero_patch() {
        assert_eq!(classify((0, 0, 0), (0, 0, 1)), BumpVerdict::Patch);
    }
}
