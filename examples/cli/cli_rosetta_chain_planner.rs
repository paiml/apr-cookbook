//! # apr rosetta — Multi-Step Conversion Chain Planner
//!
//! `apr rosetta chain <FROM> <TO>` plans the shortest conversion path
//! between two model formats by traversing the format graph (APR ↔
//! SafeTensors ↔ GGUF ↔ ONNX). This recipe builds the BFS planner as a
//! pure function so a CI pipeline can preview the chain (number of hops,
//! intermediate formats) before invoking the binary.
//!
//! Demonstrates the **ROSETTA.3** recipe for PMAT-094 (apr rosetta coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-ROSETTA-001 + format-graph spec
//!
//! Run with: cargo run --example cli_rosetta_chain_planner
//!
//! Added by PMAT-094 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Format {
    Apr,
    SafeTensors,
    Gguf,
    Onnx,
}

/// Direct conversion edges. Each edge means "X can be losslessly converted to Y".
fn edges() -> HashMap<Format, Vec<Format>> {
    let mut m = HashMap::new();
    m.insert(Format::Apr, vec![Format::SafeTensors, Format::Gguf]);
    m.insert(Format::SafeTensors, vec![Format::Apr, Format::Onnx]);
    m.insert(Format::Gguf, vec![Format::Apr]);
    m.insert(Format::Onnx, vec![Format::SafeTensors]);
    m
}

pub fn plan_chain(from: Format, to: Format) -> Option<Vec<Format>> {
    if from == to {
        return Some(vec![from]);
    }
    let g = edges();
    // BFS with parent map for path reconstruction.
    let mut parent: HashMap<Format, Format> = HashMap::new();
    let mut q: VecDeque<Format> = VecDeque::new();
    let mut seen: HashSet<Format> = HashSet::new();
    q.push_back(from);
    seen.insert(from);
    while let Some(cur) = q.pop_front() {
        if cur == to {
            // Reconstruct path
            let mut path = vec![cur];
            let mut node = cur;
            while let Some(&p) = parent.get(&node) {
                path.push(p);
                node = p;
            }
            path.reverse();
            return Some(path);
        }
        if let Some(neighbors) = g.get(&cur) {
            for &n in neighbors {
                if seen.insert(n) {
                    parent.insert(n, cur);
                    q.push_back(n);
                }
            }
        }
    }
    None
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_rosetta_chain_planner")?;

    let cases = [
        (Format::Apr, Format::Onnx),        // 3 hops
        (Format::Gguf, Format::Onnx),       // 4 hops
        (Format::Apr, Format::SafeTensors), // 2 hops (direct)
        (Format::Gguf, Format::Gguf),       // 1 hop (identity)
    ];
    for (a, b) in cases {
        let chain = plan_chain(a, b);
        println!("{a:?}  →  {b:?}  =  {chain:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn identity_is_one_hop() {
        let p = plan_chain(Format::Apr, Format::Apr).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0], Format::Apr);
    }

    #[test]
    fn direct_edge_is_two_hops() {
        // APR has a direct edge to SafeTensors → path length 2.
        let p = plan_chain(Format::Apr, Format::SafeTensors).unwrap();
        assert_eq!(p, vec![Format::Apr, Format::SafeTensors]);
    }

    #[test]
    fn apr_to_onnx_via_safetensors() {
        // No direct APR → ONNX edge; must go via SafeTensors.
        let p = plan_chain(Format::Apr, Format::Onnx).unwrap();
        assert_eq!(p, vec![Format::Apr, Format::SafeTensors, Format::Onnx]);
    }

    #[test]
    fn gguf_to_onnx_is_longest_chain() {
        // GGUF → APR → SafeTensors → ONNX (4 nodes).
        let p = plan_chain(Format::Gguf, Format::Onnx).unwrap();
        assert_eq!(p.len(), 4);
        assert_eq!(p.first(), Some(&Format::Gguf));
        assert_eq!(p.last(), Some(&Format::Onnx));
    }

    #[test]
    fn bfs_returns_shortest_path() {
        // Sanity: returned path length matches BFS shortest-path semantics.
        // SafeTensors → APR is direct; SafeTensors → ONNX → SafeTensors → APR
        // is also a valid walk but BFS must pick the direct edge.
        let p = plan_chain(Format::SafeTensors, Format::Apr).unwrap();
        assert_eq!(p.len(), 2);
    }
}
