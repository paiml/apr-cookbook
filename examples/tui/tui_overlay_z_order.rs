//! # TUI Overlay Z-Order
//!
//! Sort overlays by z-index (higher = on top). When overlays have
//! equal z-index, preserve insertion order. Returns paint order
//! bottom-to-top.
//!
//! Demonstrates the **TUI.23** recipe for PMAT-167 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HTML CSS z-index stacking context.
//!
//! Run with: cargo run --example tui_overlay_z_order
//!
//! Added by PMAT-167 (catalog 1126→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Overlay {
    pub id: String,
    pub z_index: i32,
}

#[derive(Debug, PartialEq)]
pub enum ZOrderVerdict {
    Ok { paint_order: Vec<String> },
    EmptyOverlays,
}

pub fn order(overlays: &[Overlay]) -> ZOrderVerdict {
    if overlays.is_empty() {
        return ZOrderVerdict::EmptyOverlays;
    }
    let mut indexed: Vec<(usize, &Overlay)> = overlays.iter().enumerate().collect();
    // Stable sort: preserves insertion order for equal z-index.
    indexed.sort_by(|a, b| a.1.z_index.cmp(&b.1.z_index).then(a.0.cmp(&b.0)));
    ZOrderVerdict::Ok {
        paint_order: indexed.into_iter().map(|(_, o)| o.id.clone()).collect(),
    }
}

fn ov(id: &str, z: i32) -> Overlay {
    Overlay {
        id: id.to_string(),
        z_index: z,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_overlay_z_order")?;

    let overlays = vec![ov("base", 0), ov("modal", 100), ov("toast", 50)];
    println!("typical: {:?}", order(&overlays));

    let same_z = vec![ov("a", 10), ov("b", 10), ov("c", 10)];
    println!("equal z preserves order: {:?}", order(&same_z));
    println!("empty: {:?}", order(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orderer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn higher_z_paints_later() {
        let v = order(&[ov("a", 0), ov("b", 100), ov("c", 50)]);
        if let ZOrderVerdict::Ok { paint_order } = v {
            assert_eq!(
                paint_order,
                vec!["a".to_string(), "c".to_string(), "b".to_string()]
            );
        }
    }

    #[test]
    fn equal_z_preserves_insertion() {
        let v = order(&[ov("a", 10), ov("b", 10), ov("c", 10)]);
        if let ZOrderVerdict::Ok { paint_order } = v {
            assert_eq!(
                paint_order,
                vec!["a".to_string(), "b".to_string(), "c".to_string()]
            );
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(order(&[]), ZOrderVerdict::EmptyOverlays);
    }

    #[test]
    fn single_overlay_returned() {
        let v = order(&[ov("only", 0)]);
        if let ZOrderVerdict::Ok { paint_order } = v {
            assert_eq!(paint_order, vec!["only".to_string()]);
        }
    }

    #[test]
    fn negative_z_supported() {
        let v = order(&[ov("a", 0), ov("below", -100)]);
        if let ZOrderVerdict::Ok { paint_order } = v {
            assert_eq!(paint_order[0], "below");
        }
    }

    #[test]
    fn many_overlays() {
        let overlays: Vec<Overlay> = (0..20).map(|i| ov(&format!("o{i}"), i as i32)).collect();
        let v = order(&overlays);
        if let ZOrderVerdict::Ok { paint_order } = v {
            assert_eq!(paint_order.len(), 20);
            assert_eq!(paint_order[0], "o0");
            assert_eq!(paint_order[19], "o19");
        }
    }

    #[test]
    fn duplicates_in_input() {
        let v = order(&[ov("dup", 5), ov("dup", 5)]);
        if let ZOrderVerdict::Ok { paint_order } = v {
            assert_eq!(paint_order.len(), 2);
        }
    }

    #[test]
    fn extreme_z_values() {
        let v = order(&[ov("min", i32::MIN), ov("max", i32::MAX)]);
        if let ZOrderVerdict::Ok { paint_order } = v {
            assert_eq!(paint_order[0], "min");
            assert_eq!(paint_order[1], "max");
        }
    }

    #[test]
    fn unicode_ids() {
        let v = order(&[ov("café", 5), ov("résumé", 10)]);
        if let ZOrderVerdict::Ok { paint_order } = v {
            assert_eq!(paint_order[0], "café");
        }
    }

    #[test]
    fn deterministic() {
        let overlays = vec![ov("a", 0), ov("b", 100)];
        let a = order(&overlays);
        let b = order(&overlays);
        assert_eq!(a, b);
    }
}
