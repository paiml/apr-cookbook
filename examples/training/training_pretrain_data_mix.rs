//! # Training Pretrain Data Mix Picker
//!
//! Compose pretraining mixture: code/web/wikipedia/arxiv ratios.
//! Llama-2 used roughly 4.5T tokens (~code 5%, common-crawl 67%, wiki
//! 4.5%, books 4.5%, arxiv 2.5%, stack-exchange 2%, github 4%).
//!
//! Picker: given target task domain (general/code/scientific), returns
//! mixture ratios.
//!
//! Demonstrates the **TRAIN.17** recipe for PMAT-146 (training round 5).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Llama 2 paper (Touvron et al., 2023) data composition.
//!
//! Run with: cargo run --example training_pretrain_data_mix
//!
//! Added by PMAT-146 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetDomain {
    GeneralPurpose,
    CodeFocused,
    Scientific,
    Multilingual,
}

#[derive(Debug, PartialEq)]
pub enum MixVerdict {
    Ok {
        web_pct: u32,
        code_pct: u32,
        wiki_pct: u32,
        arxiv_pct: u32,
        books_pct: u32,
    },
}

pub fn pick(domain: TargetDomain) -> MixVerdict {
    match domain {
        TargetDomain::GeneralPurpose => MixVerdict::Ok {
            web_pct: 70,
            code_pct: 8,
            wiki_pct: 8,
            arxiv_pct: 4,
            books_pct: 10,
        },
        TargetDomain::CodeFocused => MixVerdict::Ok {
            web_pct: 30,
            code_pct: 50,
            wiki_pct: 5,
            arxiv_pct: 5,
            books_pct: 10,
        },
        TargetDomain::Scientific => MixVerdict::Ok {
            web_pct: 30,
            code_pct: 5,
            wiki_pct: 15,
            arxiv_pct: 35,
            books_pct: 15,
        },
        TargetDomain::Multilingual => MixVerdict::Ok {
            web_pct: 80,
            code_pct: 5,
            wiki_pct: 10,
            arxiv_pct: 2,
            books_pct: 3,
        },
    }
}

pub fn sums_to_100(verdict: &MixVerdict) -> bool {
    let MixVerdict::Ok {
        web_pct,
        code_pct,
        wiki_pct,
        arxiv_pct,
        books_pct,
    } = verdict;
    web_pct + code_pct + wiki_pct + arxiv_pct + books_pct == 100
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("training_pretrain_data_mix")?;

    for domain in [
        TargetDomain::GeneralPurpose,
        TargetDomain::CodeFocused,
        TargetDomain::Scientific,
        TargetDomain::Multilingual,
    ] {
        println!("{domain:?}: {:?}", pick(domain));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn general_sums_to_100() {
        let v = pick(TargetDomain::GeneralPurpose);
        assert!(sums_to_100(&v));
    }

    #[test]
    fn code_sums_to_100() {
        let v = pick(TargetDomain::CodeFocused);
        assert!(sums_to_100(&v));
    }

    #[test]
    fn scientific_sums_to_100() {
        let v = pick(TargetDomain::Scientific);
        assert!(sums_to_100(&v));
    }

    #[test]
    fn multilingual_sums_to_100() {
        let v = pick(TargetDomain::Multilingual);
        assert!(sums_to_100(&v));
    }

    #[test]
    fn code_focused_has_high_code_share() {
        if let MixVerdict::Ok { code_pct, .. } = pick(TargetDomain::CodeFocused) {
            assert!(code_pct >= 40);
        }
    }

    #[test]
    fn scientific_has_high_arxiv_share() {
        if let MixVerdict::Ok { arxiv_pct, .. } = pick(TargetDomain::Scientific) {
            assert!(arxiv_pct >= 25);
        }
    }

    #[test]
    fn general_dominated_by_web() {
        if let MixVerdict::Ok {
            web_pct, code_pct, ..
        } = pick(TargetDomain::GeneralPurpose)
        {
            assert!(web_pct > code_pct);
        }
    }

    #[test]
    fn multilingual_largest_web_share() {
        let multi = pick(TargetDomain::Multilingual);
        let general = pick(TargetDomain::GeneralPurpose);
        if let (MixVerdict::Ok { web_pct: m, .. }, MixVerdict::Ok { web_pct: g, .. }) =
            (multi, general)
        {
            assert!(m >= g);
        }
    }

    #[test]
    fn code_pct_zero_or_more() {
        for d in [
            TargetDomain::GeneralPurpose,
            TargetDomain::CodeFocused,
            TargetDomain::Scientific,
            TargetDomain::Multilingual,
        ] {
            let v = pick(d);
            if let MixVerdict::Ok { code_pct, .. } = v {
                assert!(code_pct < 100);
            }
        }
    }

    #[test]
    fn each_domain_unique_mix() {
        let general = pick(TargetDomain::GeneralPurpose);
        let code = pick(TargetDomain::CodeFocused);
        let sci = pick(TargetDomain::Scientific);
        assert_ne!(general, code);
        assert_ne!(code, sci);
        assert_ne!(general, sci);
    }
}
