//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use clap::Parser;
use proptest::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Parser)]
#[command(name = "apr-runs", about = "List, show, and compare training runs")]
pub struct RunsConfig {
    /// Subcommand: list, show, compare
    #[arg(default_value = "list")]
    pub subcommand: String,

    // First run ID (for show/compare)
    pub run_id: Option<String>,

    // Second run ID (for compare)
    pub compare_id: Option<String>,

    /// Generate demo training runs
    #[arg(long, short)]
    pub demo: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRun {
    pub id: String,
    pub name: String,
    pub epoch: u32,
    pub loss: f64,
    pub accuracy: f64,
    pub duration_secs: u64,
    pub timestamp: String,
    pub hyperparams: HashMap<String, String>,
}
