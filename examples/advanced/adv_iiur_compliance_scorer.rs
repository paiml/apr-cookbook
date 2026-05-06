//! # Advanced IIUR Compliance Scorer
//!
//! IIUR principles: Isolated, Idempotent, Useful, Reproducible. A
//! recipe scores 1 point per principle satisfied. This recipe builds
//! the per-principle checker over a recipe-source string + a tier
//! classifier (4=Gold/3=Silver/2=Bronze/<2=Fail).
//!
//! Demonstrates the **ADV.4** recipe for PMAT-128 (advanced coverage —
//! closing F-invariant gap).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender IIUR-001.
//!
//! Run with: cargo run --example adv_iiur_compliance_scorer
//!
//! Added by PMAT-128 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum IiurTier {
    Gold,   // 4/4
    Silver, // 3/4
    Bronze, // 2/4
    Fail,   // < 2/4
}

#[derive(Debug, PartialEq)]
pub struct IiurScore {
    pub isolated: bool,
    pub idempotent: bool,
    pub useful: bool,
    pub reproducible: bool,
    pub tier: IiurTier,
}

pub fn score(source: &str) -> IiurScore {
    let isolated = !source.contains("std::env::set_var")
        && !source.contains("std::process::exit")
        && !source.contains("static mut ");
    let idempotent = !source.contains("std::time::Instant::now()")
        && !source.contains("rand::thread_rng()")
        && !source.contains("std::time::SystemTime::now()");
    let useful = source.contains("#[cfg(test)]") && source.contains("#[test]");
    let reproducible = source.contains("RecipeContext::new");
    let count = [isolated, idempotent, useful, reproducible]
        .iter()
        .filter(|x| **x)
        .count();
    let tier = match count {
        4 => IiurTier::Gold,
        3 => IiurTier::Silver,
        2 => IiurTier::Bronze,
        _ => IiurTier::Fail,
    };
    IiurScore {
        isolated,
        idempotent,
        useful,
        reproducible,
        tier,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_iiur_compliance_scorer")?;

    let gold = r#"
        use apr_cookbook::recipe::RecipeContext;
        fn main() { let _ctx = RecipeContext::new("test"); }
        #[cfg(test)] mod tests { #[test] fn t() {} }
    "#;
    let silver = "use apr_cookbook::recipe::RecipeContext; #[cfg(test)] #[test] fn t() {}";
    let bronze = "use apr_cookbook::recipe::RecipeContext; fn main() {}";
    let fail = "fn main() { rand::thread_rng(); std::env::set_var(\"X\", \"1\"); }";

    println!("gold:   {:?}", score(gold));
    println!("silver: {:?}", score(silver));
    println!("bronze: {:?}", score(bronze));
    println!("fail:   {:?}", score(fail));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scorer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn full_compliance_gold() {
        let s = r#"
            use apr_cookbook::recipe::RecipeContext;
            fn main() { let _ctx = RecipeContext::new("t"); }
            #[cfg(test)] mod tests { #[test] fn t() {} }
        "#;
        let r = score(s);
        assert_eq!(r.tier, IiurTier::Gold);
        assert!(r.isolated && r.idempotent && r.useful && r.reproducible);
    }

    #[test]
    fn missing_tests_drops_to_silver() {
        let s = "use apr_cookbook::recipe::RecipeContext; fn main() { let _ = RecipeContext::new(\"t\"); }";
        let r = score(s);
        assert_eq!(r.tier, IiurTier::Silver);
        assert!(!r.useful);
    }

    #[test]
    fn unsafe_static_breaks_isolation() {
        let s = r#"
            use apr_cookbook::recipe::RecipeContext;
            static mut COUNTER: u32 = 0;
            fn main() { let _ = RecipeContext::new("t"); }
            #[cfg(test)] #[test] fn t() {}
        "#;
        let r = score(s);
        assert!(!r.isolated);
        assert_eq!(r.tier, IiurTier::Silver);
    }

    #[test]
    fn rand_thread_rng_breaks_idempotence() {
        let s = r#"
            use apr_cookbook::recipe::RecipeContext;
            fn main() { rand::thread_rng(); let _ = RecipeContext::new("t"); }
            #[cfg(test)] #[test] fn t() {}
        "#;
        let r = score(s);
        assert!(!r.idempotent);
    }

    #[test]
    fn instant_now_breaks_idempotence() {
        let s = "use apr_cookbook::recipe::RecipeContext; let _t = std::time::Instant::now(); #[test] fn t() {}";
        let r = score(s);
        assert!(!r.idempotent);
    }

    #[test]
    fn no_recipe_context_breaks_reproducibility() {
        let s = "fn main() {} #[cfg(test)] #[test] fn t() {}";
        let r = score(s);
        assert!(!r.reproducible);
    }

    #[test]
    fn process_exit_breaks_isolation() {
        let s = "use apr_cookbook::recipe::RecipeContext; fn main() { std::process::exit(1); }";
        let r = score(s);
        assert!(!r.isolated);
    }

    #[test]
    fn empty_source_scores_isolated_idempotent_only() {
        // Empty has no anti-patterns → isolated=true + idempotent=true,
        // but no #[test] (useful=false) and no RecipeContext (reproducible=false)
        // → 2/4 = Bronze.
        let r = score("");
        assert_eq!(r.tier, IiurTier::Bronze);
        assert!(r.isolated && r.idempotent);
        assert!(!r.useful && !r.reproducible);
    }

    #[test]
    fn fail_tier_when_anti_patterns_dominate() {
        // Two anti-patterns (rand + static mut) + no test + no RecipeContext → 0/4.
        let s = "static mut X: u32 = 0; fn main() { rand::thread_rng(); }";
        let r = score(s);
        assert_eq!(r.tier, IiurTier::Fail);
    }
}
