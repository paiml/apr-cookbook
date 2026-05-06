//! # apr gpu --peer-access — Topology Validator
//!
//! P2P GPU access depends on the bus topology. NVLink-connected pairs
//! support direct memory access; PCIe-only pairs require a host
//! intermediate (slower). This recipe builds the validator over an
//! adjacency matrix + classifies the configuration.
//!
//! Demonstrates the **GPU.5** recipe for PMAT-120 (apr gpu coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender GPU-001 + NVIDIA NVLink Topology
//!
//! Run with: cargo run --example cli_gpu_peer_access_topology
//!
//! Added by PMAT-120 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkKind {
    NvLink,
    Pcie,
    None,
}

#[derive(Debug, PartialEq)]
pub enum TopologyVerdict {
    AllNvLink,
    Mixed {
        nvlink_pairs: usize,
        pcie_pairs: usize,
    },
    AllPcie,
    Disconnected {
        isolated_gpus: Vec<usize>,
    },
    InvalidShape,
}

#[allow(clippy::needless_range_loop)]
pub fn classify(links: &[Vec<LinkKind>]) -> TopologyVerdict {
    let n = links.len();
    if n == 0 {
        return TopologyVerdict::InvalidShape;
    }
    if links.iter().any(|row| row.len() != n) {
        return TopologyVerdict::InvalidShape;
    }
    // Diagonal must be None (self-pair).
    for (i, row) in links.iter().enumerate() {
        if row[i] != LinkKind::None {
            return TopologyVerdict::InvalidShape;
        }
    }
    let mut nvlink_pairs = 0usize;
    let mut pcie_pairs = 0usize;
    let mut isolated = Vec::new();
    for i in 0..n {
        let mut has_link = false;
        for j in (i + 1)..n {
            match links[i][j] {
                LinkKind::NvLink => {
                    nvlink_pairs += 1;
                    has_link = true;
                }
                LinkKind::Pcie => {
                    pcie_pairs += 1;
                    has_link = true;
                }
                LinkKind::None => {}
            }
            if links[j][i] != LinkKind::None {
                has_link = true;
            }
        }
        // Also check incoming edges.
        if !has_link
            && (0..i).all(|prev| links[prev][i] == LinkKind::None)
            && (0..n).all(|other| other == i || links[i][other] == LinkKind::None)
        {
            isolated.push(i);
        }
    }
    if !isolated.is_empty() {
        return TopologyVerdict::Disconnected {
            isolated_gpus: isolated,
        };
    }
    match (nvlink_pairs, pcie_pairs) {
        (n, 0) if n > 0 => TopologyVerdict::AllNvLink,
        (0, p) if p > 0 => TopologyVerdict::AllPcie,
        _ => TopologyVerdict::Mixed {
            nvlink_pairs,
            pcie_pairs,
        },
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_gpu_peer_access_topology")?;

    let nvlink_4x = vec![
        vec![
            LinkKind::None,
            LinkKind::NvLink,
            LinkKind::NvLink,
            LinkKind::NvLink,
        ],
        vec![
            LinkKind::NvLink,
            LinkKind::None,
            LinkKind::NvLink,
            LinkKind::NvLink,
        ],
        vec![
            LinkKind::NvLink,
            LinkKind::NvLink,
            LinkKind::None,
            LinkKind::NvLink,
        ],
        vec![
            LinkKind::NvLink,
            LinkKind::NvLink,
            LinkKind::NvLink,
            LinkKind::None,
        ],
    ];
    println!("nvlink 4x: {:?}", classify(&nvlink_4x));

    let mixed = vec![
        vec![LinkKind::None, LinkKind::NvLink, LinkKind::Pcie],
        vec![LinkKind::NvLink, LinkKind::None, LinkKind::Pcie],
        vec![LinkKind::Pcie, LinkKind::Pcie, LinkKind::None],
    ];
    println!("mixed:     {:?}", classify(&mixed));
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
    fn all_nvlink_classified() {
        let m = vec![
            vec![LinkKind::None, LinkKind::NvLink],
            vec![LinkKind::NvLink, LinkKind::None],
        ];
        assert_eq!(classify(&m), TopologyVerdict::AllNvLink);
    }

    #[test]
    fn all_pcie_classified() {
        let m = vec![
            vec![LinkKind::None, LinkKind::Pcie],
            vec![LinkKind::Pcie, LinkKind::None],
        ];
        assert_eq!(classify(&m), TopologyVerdict::AllPcie);
    }

    #[test]
    fn mixed_classified() {
        let m = vec![
            vec![LinkKind::None, LinkKind::NvLink, LinkKind::Pcie],
            vec![LinkKind::NvLink, LinkKind::None, LinkKind::Pcie],
            vec![LinkKind::Pcie, LinkKind::Pcie, LinkKind::None],
        ];
        let v = classify(&m);
        assert!(matches!(v, TopologyVerdict::Mixed { .. }));
    }

    #[test]
    fn empty_invalid() {
        assert_eq!(classify(&[]), TopologyVerdict::InvalidShape);
    }

    #[test]
    fn non_square_invalid() {
        let m = vec![
            vec![LinkKind::None, LinkKind::NvLink],
            vec![LinkKind::NvLink],
        ];
        assert_eq!(classify(&m), TopologyVerdict::InvalidShape);
    }

    #[test]
    fn diagonal_must_be_none() {
        let m = vec![
            vec![LinkKind::NvLink, LinkKind::NvLink],
            vec![LinkKind::NvLink, LinkKind::None],
        ];
        assert_eq!(classify(&m), TopologyVerdict::InvalidShape);
    }

    #[test]
    fn isolated_gpu_detected() {
        // GPU 2 has no peers.
        let m = vec![
            vec![LinkKind::None, LinkKind::NvLink, LinkKind::None],
            vec![LinkKind::NvLink, LinkKind::None, LinkKind::None],
            vec![LinkKind::None, LinkKind::None, LinkKind::None],
        ];
        let v = classify(&m);
        assert!(matches!(v, TopologyVerdict::Disconnected { .. }));
    }

    #[test]
    fn single_gpu_topology_with_no_peers() {
        // One GPU is technically OK — no peer pairs to classify.
        let m = vec![vec![LinkKind::None]];
        let v = classify(&m);
        assert!(matches!(v, TopologyVerdict::Disconnected { .. }));
    }
}
