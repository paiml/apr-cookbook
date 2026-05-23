//! # apr publish — `--dry-run` Local-vs-Hub File Diff
//!
//! `apr publish ... --dry-run` previews which files would be uploaded
//! without actually contacting the Hub. The preview must show three
//! categories of action: NEW (file in local, not in hub), UPDATED (file
//! in both, content hash differs), UNCHANGED (file in both, hash matches).
//! This recipe builds the diff classifier as a pure function.
//!
//! Demonstrates the **PUBLISH.9** recipe for PMAT-098 (apr publish coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender APR-PUB-001 + content-addressed upload protocol
//!
//! Run with: cargo run --example cli_publish_dry_run_diff
//!
//! Added by PMAT-098 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffAction {
    New,
    Updated {
        local_hash: String,
        hub_hash: String,
    },
    Unchanged,
    HubOnly, // exists on hub, not local — won't be deleted on dry-run
}

pub fn diff_local_vs_hub(
    local: &BTreeMap<String, String>, // path → content_hash
    hub: &BTreeMap<String, String>,
) -> BTreeMap<String, DiffAction> {
    let mut out: BTreeMap<String, DiffAction> = BTreeMap::new();
    for (path, lhash) in local {
        match hub.get(path) {
            None => {
                out.insert(path.clone(), DiffAction::New);
            }
            Some(hhash) if hhash == lhash => {
                out.insert(path.clone(), DiffAction::Unchanged);
            }
            Some(hhash) => {
                out.insert(
                    path.clone(),
                    DiffAction::Updated {
                        local_hash: lhash.clone(),
                        hub_hash: hhash.clone(),
                    },
                );
            }
        }
    }
    for path in hub.keys() {
        if !local.contains_key(path) {
            out.insert(path.clone(), DiffAction::HubOnly);
        }
    }
    out
}

pub fn count_uploads(diff: &BTreeMap<String, DiffAction>) -> usize {
    diff.values()
        .filter(|a| matches!(a, DiffAction::New | DiffAction::Updated { .. }))
        .count()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_publish_dry_run_diff")?;

    let mut local: BTreeMap<String, String> = BTreeMap::new();
    local.insert("model.apr".into(), "h1".into());
    local.insert("README.md".into(), "h2".into());
    local.insert("tokenizer.json".into(), "h3".into());

    let mut hub: BTreeMap<String, String> = BTreeMap::new();
    hub.insert("model.apr".into(), "h1-OLD".into()); // updated
    hub.insert("README.md".into(), "h2".into()); // unchanged
    hub.insert("config.json".into(), "h4".into()); // hub-only

    let diff = diff_local_vs_hub(&local, &hub);
    for (path, action) in &diff {
        println!("  {path:>20}  {action:?}");
    }
    println!("\nDry-run uploads: {}", count_uploads(&diff));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_runs() {
        main().expect("recipe execution failed");
    }

    fn map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    #[test]
    fn new_file_classified_as_new() {
        let local = map(&[("a", "h1")]);
        let hub = map(&[]);
        let d = diff_local_vs_hub(&local, &hub);
        assert_eq!(d.get("a"), Some(&DiffAction::New));
    }

    #[test]
    fn matching_hash_is_unchanged() {
        let local = map(&[("a", "h1")]);
        let hub = map(&[("a", "h1")]);
        let d = diff_local_vs_hub(&local, &hub);
        assert_eq!(d.get("a"), Some(&DiffAction::Unchanged));
    }

    #[test]
    fn different_hash_is_updated() {
        let local = map(&[("a", "h1")]);
        let hub = map(&[("a", "h0")]);
        let d = diff_local_vs_hub(&local, &hub);
        match d.get("a") {
            Some(DiffAction::Updated {
                local_hash,
                hub_hash,
            }) => {
                assert_eq!(local_hash, "h1");
                assert_eq!(hub_hash, "h0");
            }
            other => panic!("expected Updated, got {other:?}"),
        }
    }

    #[test]
    fn hub_only_file_listed_separately() {
        let local = map(&[("a", "h1")]);
        let hub = map(&[("a", "h1"), ("b", "h2")]);
        let d = diff_local_vs_hub(&local, &hub);
        assert_eq!(d.get("b"), Some(&DiffAction::HubOnly));
    }

    #[test]
    fn upload_count_excludes_unchanged_and_hub_only() {
        let local = map(&[("a", "h1"), ("b", "h2"), ("c", "h3")]);
        let hub = map(&[("a", "h1"), ("b", "OLD"), ("d", "h4")]);
        let d = diff_local_vs_hub(&local, &hub);
        // a unchanged + b updated + c new + d hub-only = 2 uploads (b, c).
        assert_eq!(count_uploads(&d), 2);
    }

    #[test]
    fn empty_local_yields_only_hub_only_entries() {
        let local: BTreeMap<String, String> = BTreeMap::new();
        let hub = map(&[("a", "h1"), ("b", "h2")]);
        let d = diff_local_vs_hub(&local, &hub);
        for v in d.values() {
            assert_eq!(v, &DiffAction::HubOnly);
        }
    }

    #[test]
    fn empty_both_yields_empty_diff() {
        let d = diff_local_vs_hub(&BTreeMap::new(), &BTreeMap::new());
        assert!(d.is_empty());
    }
}
