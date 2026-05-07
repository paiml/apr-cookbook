//! # TUI Tag Cloud Size Compute
//!
//! Compute display font-size for each tag based on its frequency vs
//! the corpus. Returns sorted (tag, font_size_pt) pairs and the size
//! range used.
//!
//! Demonstrates the **TUI.162** recipe for PMAT-213 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Hearst & Rosner, "Tag Clouds: Data Analysis Tool"
//!  HICSS (2008); WordPress tag-cloud weight algorithm.
//!
//! Run with: cargo run --example tui_tag_cloud_size_compute
//!
//! Added by PMAT-213 (catalog 1540→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TagCloudVerdict {
    Ok {
        sized_tags: Vec<(String, u32)>,
        min_size: u32,
        max_size: u32,
    },
    InvalidConfig,
}

/// Items: (tag, frequency). min/max font-size for the cloud.
pub fn compute(items: &[(&str, u32)], min_size: u32, max_size: u32) -> TagCloudVerdict {
    if items.is_empty() || min_size >= max_size || min_size == 0 {
        return TagCloudVerdict::InvalidConfig;
    }
    for (_, freq) in items {
        if *freq == 0 {
            return TagCloudVerdict::InvalidConfig;
        }
    }
    let max_freq = items.iter().map(|(_, f)| *f).max().unwrap_or(1);
    let min_freq = items.iter().map(|(_, f)| *f).min().unwrap_or(1);
    let mut sized: Vec<(String, u32)> = items
        .iter()
        .map(|(tag, freq)| {
            let size = if max_freq == min_freq {
                (min_size + max_size) / 2
            } else {
                let pct = (freq - min_freq) as f64 / (max_freq - min_freq) as f64;
                (min_size as f64 + pct * (max_size - min_size) as f64) as u32
            };
            ((*tag).to_string(), size)
        })
        .collect();
    sized.sort();
    TagCloudVerdict::Ok {
        sized_tags: sized,
        min_size,
        max_size,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_tag_cloud_size_compute")?;

    let items = [("rust", 50), ("python", 100), ("c", 20)];
    println!("cloud: {:?}", compute(&items, 10, 30));
    println!("invalid: {:?}", compute(&[], 10, 30));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(compute(&[], 10, 30), TagCloudVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_min_ge_max() {
        let items = [("a", 5)];
        assert_eq!(compute(&items, 30, 10), TagCloudVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_min() {
        let items = [("a", 5)];
        assert_eq!(compute(&items, 0, 30), TagCloudVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_freq() {
        let items = [("a", 0)];
        assert_eq!(compute(&items, 10, 30), TagCloudVerdict::InvalidConfig);
    }

    #[test]
    fn highest_freq_max_size() {
        let items = [("low", 10), ("high", 100)];
        let v = compute(&items, 10, 30);
        if let TagCloudVerdict::Ok { sized_tags, .. } = v {
            let high_size = sized_tags.iter().find(|(t, _)| t == "high").unwrap().1;
            assert_eq!(high_size, 30);
        }
    }

    #[test]
    fn lowest_freq_min_size() {
        let items = [("low", 10), ("high", 100)];
        let v = compute(&items, 10, 30);
        if let TagCloudVerdict::Ok { sized_tags, .. } = v {
            let low_size = sized_tags.iter().find(|(t, _)| t == "low").unwrap().1;
            assert_eq!(low_size, 10);
        }
    }

    #[test]
    fn equal_freqs_midpoint() {
        let items = [("a", 50), ("b", 50)];
        let v = compute(&items, 10, 30);
        if let TagCloudVerdict::Ok { sized_tags, .. } = v {
            for (_, s) in &sized_tags {
                assert_eq!(*s, 20);
            }
        }
    }

    #[test]
    fn deterministic() {
        let items = [("a", 50)];
        let r1 = compute(&items, 10, 30);
        let r2 = compute(&items, 10, 30);
        assert_eq!(r1, r2);
    }

    #[test]
    fn tags_sorted() {
        let items = [("zeta", 50), ("alpha", 50)];
        let v = compute(&items, 10, 30);
        if let TagCloudVerdict::Ok { sized_tags, .. } = v {
            assert_eq!(sized_tags[0].0, "alpha");
            assert_eq!(sized_tags[1].0, "zeta");
        }
    }

    #[test]
    fn many_tags_handled() {
        let items: Vec<(&str, u32)> = (0..30).map(|_| ("t", 50)).collect();
        let v = compute(&items, 10, 30);
        if let TagCloudVerdict::Ok { sized_tags, .. } = v {
            assert_eq!(sized_tags.len(), 30);
        }
    }

    #[test]
    fn unicode_tag_supported() {
        let items = [("café", 50)];
        let v = compute(&items, 10, 30);
        if let TagCloudVerdict::Ok { sized_tags, .. } = v {
            assert_eq!(sized_tags[0].0, "café");
        }
    }

    #[test]
    fn size_in_min_max_range() {
        let items = [("a", 50), ("b", 100), ("c", 25)];
        let v = compute(&items, 10, 30);
        if let TagCloudVerdict::Ok { sized_tags, .. } = v {
            for (_, s) in &sized_tags {
                assert!(*s >= 10 && *s <= 30);
            }
        }
    }
}
