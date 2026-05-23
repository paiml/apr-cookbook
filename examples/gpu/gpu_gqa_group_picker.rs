//! # GPU Grouped-Query-Attention Group Picker
//!
//! GQA reduces KV memory: instead of N_heads K/V heads, share groups
//! of N/G KV heads across queries. Llama-2 70B used 8 (down from 64
//! query heads → 8 KV groups).
//!
//! Picker rules:
//!   group_size = max(1, n_heads / target_groups), with target = 8 typical
//!   memory_savings = 1 - groups / n_heads
//!
//! Demonstrates the **GPU.30** recipe for PMAT-146 (gpu round 5).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Ainslie et al. (2023). GQA: Generalized Grouped-Query Attention.
//!
//! Run with: cargo run --example gpu_gqa_group_picker
//!
//! Added by PMAT-146 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum GqaVerdict {
    Ok {
        n_kv_groups: u32,
        group_size: u32,
        kv_memory_savings_pct: u32,
    },
    InvalidHeadCount,
    NotDivisible {
        n_heads: u32,
        target: u32,
    },
}

pub fn pick(n_query_heads: u32, target_kv_groups: u32) -> GqaVerdict {
    if n_query_heads == 0 || target_kv_groups == 0 {
        return GqaVerdict::InvalidHeadCount;
    }
    if target_kv_groups > n_query_heads {
        return GqaVerdict::NotDivisible {
            n_heads: n_query_heads,
            target: target_kv_groups,
        };
    }
    if n_query_heads % target_kv_groups != 0 {
        return GqaVerdict::NotDivisible {
            n_heads: n_query_heads,
            target: target_kv_groups,
        };
    }
    let group_size = n_query_heads / target_kv_groups;
    let savings = 100 - (target_kv_groups * 100) / n_query_heads;
    GqaVerdict::Ok {
        n_kv_groups: target_kv_groups,
        group_size,
        kv_memory_savings_pct: savings,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("gpu_gqa_group_picker")?;

    println!("Llama-3 70B (64H, 8G): {:?}", pick(64, 8));
    println!("Llama-3 8B (32H, 8G): {:?}", pick(32, 8));
    println!("Mistral 7B (32H, 8G): {:?}", pick(32, 8));
    println!("MQA (32H, 1G): {:?}", pick(32, 1));
    println!("MHA (32H, 32G): {:?}", pick(32, 32));
    println!("not divisible: {:?}", pick(33, 8));
    println!("invalid: {:?}", pick(0, 8));
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
    fn llama_70b_typical() {
        let v = pick(64, 8);
        if let GqaVerdict::Ok {
            n_kv_groups,
            group_size,
            ..
        } = v
        {
            assert_eq!(n_kv_groups, 8);
            assert_eq!(group_size, 8);
        }
    }

    #[test]
    fn savings_proportional_to_group_count() {
        // 32 heads, 1 group = MQA: savings ≈ 96%.
        if let GqaVerdict::Ok {
            kv_memory_savings_pct,
            ..
        } = pick(32, 1)
        {
            assert!(kv_memory_savings_pct >= 96);
        }
    }

    #[test]
    fn mha_no_savings() {
        // 32 heads, 32 groups = MHA: savings 0.
        if let GqaVerdict::Ok {
            kv_memory_savings_pct,
            ..
        } = pick(32, 32)
        {
            assert_eq!(kv_memory_savings_pct, 0);
        }
    }

    #[test]
    fn group_size_correct() {
        // 32 heads / 4 groups = 8.
        if let GqaVerdict::Ok { group_size, .. } = pick(32, 4) {
            assert_eq!(group_size, 8);
        }
    }

    #[test]
    fn not_divisible_rejected() {
        let v = pick(33, 8);
        assert!(matches!(v, GqaVerdict::NotDivisible { .. }));
    }

    #[test]
    fn target_above_heads_rejected() {
        let v = pick(8, 32);
        assert!(matches!(v, GqaVerdict::NotDivisible { .. }));
    }

    #[test]
    fn invalid_zero_heads() {
        assert_eq!(pick(0, 8), GqaVerdict::InvalidHeadCount);
    }

    #[test]
    fn invalid_zero_target() {
        assert_eq!(pick(32, 0), GqaVerdict::InvalidHeadCount);
    }

    #[test]
    fn mqa_single_group() {
        // 1 KV group = MQA.
        let v = pick(32, 1);
        if let GqaVerdict::Ok { n_kv_groups, .. } = v {
            assert_eq!(n_kv_groups, 1);
        }
    }

    #[test]
    fn fewer_groups_more_savings() {
        let v_8 = pick(32, 8);
        let v_4 = pick(32, 4);
        if let (
            GqaVerdict::Ok {
                kv_memory_savings_pct: s8,
                ..
            },
            GqaVerdict::Ok {
                kv_memory_savings_pct: s4,
                ..
            },
        ) = (v_8, v_4)
        {
            assert!(s4 > s8);
        }
    }
}
