//! # MCP JSON-RPC Request-ID Correlator
//!
//! JSON-RPC 2.0 (the wire format MCP rides on) pairs a request `id`
//! with exactly one response `id`. This recipe builds the in-flight
//! correlator: register on send, drain on receive, time-out stale ids.
//!
//! Demonstrates the **MCP.14** recipe for PMAT-135 (mcp coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: JSON-RPC 2.0 specification § 4 (Request and Response objects).
//!
//! Run with: cargo run --example mcp_request_id_correlator
//!
//! Added by PMAT-135 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::HashMap;

#[derive(Debug, PartialEq)]
pub enum RegisterVerdict {
    Ok,
    DuplicateId(u64),
    InvalidId,
}

#[derive(Debug, PartialEq)]
pub enum DrainVerdict {
    Matched { latency_ms: u64 },
    UnknownId,
    InvalidId,
}

pub struct Correlator {
    inflight: HashMap<u64, u64>,
    timeout_ms: u64,
}

impl Correlator {
    pub fn new(timeout_ms: u64) -> Self {
        Self {
            inflight: HashMap::new(),
            timeout_ms,
        }
    }

    pub fn register(&mut self, id: u64, sent_at_ms: u64) -> RegisterVerdict {
        if id == 0 {
            return RegisterVerdict::InvalidId;
        }
        if self.inflight.contains_key(&id) {
            return RegisterVerdict::DuplicateId(id);
        }
        self.inflight.insert(id, sent_at_ms);
        RegisterVerdict::Ok
    }

    pub fn drain(&mut self, id: u64, received_at_ms: u64) -> DrainVerdict {
        if id == 0 {
            return DrainVerdict::InvalidId;
        }
        match self.inflight.remove(&id) {
            Some(sent_at_ms) => DrainVerdict::Matched {
                latency_ms: received_at_ms.saturating_sub(sent_at_ms),
            },
            None => DrainVerdict::UnknownId,
        }
    }

    pub fn sweep_timeouts(&mut self, now_ms: u64) -> Vec<u64> {
        let cutoff = now_ms.saturating_sub(self.timeout_ms);
        let stale: Vec<u64> = self
            .inflight
            .iter()
            .filter(|(_, &sent)| sent < cutoff)
            .map(|(id, _)| *id)
            .collect();
        for id in &stale {
            self.inflight.remove(id);
        }
        stale
    }

    pub fn pending_count(&self) -> usize {
        self.inflight.len()
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mcp_request_id_correlator")?;

    let mut c = Correlator::new(5000);
    println!("register 1@100: {:?}", c.register(1, 100));
    println!("register 2@200: {:?}", c.register(2, 200));
    println!("dup 1@300: {:?}", c.register(1, 300));
    println!("invalid 0: {:?}", c.register(0, 400));
    println!("drain 1@500: {:?}", c.drain(1, 500));
    println!("drain unknown 99: {:?}", c.drain(99, 600));
    println!("pending: {}", c.pending_count());
    println!("sweep at 6000: {:?}", c.sweep_timeouts(6000));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correlator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn register_typical_succeeds() {
        let mut c = Correlator::new(1000);
        assert_eq!(c.register(1, 100), RegisterVerdict::Ok);
    }

    #[test]
    fn duplicate_id_rejected() {
        let mut c = Correlator::new(1000);
        c.register(1, 100);
        assert_eq!(c.register(1, 200), RegisterVerdict::DuplicateId(1));
    }

    #[test]
    fn id_zero_invalid() {
        let mut c = Correlator::new(1000);
        assert_eq!(c.register(0, 100), RegisterVerdict::InvalidId);
        assert_eq!(c.drain(0, 100), DrainVerdict::InvalidId);
    }

    #[test]
    fn drain_match_returns_latency() {
        let mut c = Correlator::new(1000);
        c.register(1, 100);
        let v = c.drain(1, 250);
        assert_eq!(v, DrainVerdict::Matched { latency_ms: 150 });
    }

    #[test]
    fn drain_unknown_id() {
        let mut c = Correlator::new(1000);
        assert_eq!(c.drain(99, 100), DrainVerdict::UnknownId);
    }

    #[test]
    fn drain_removes_from_inflight() {
        let mut c = Correlator::new(1000);
        c.register(1, 100);
        assert_eq!(c.pending_count(), 1);
        c.drain(1, 200);
        assert_eq!(c.pending_count(), 0);
    }

    #[test]
    fn sweep_returns_stale_ids() {
        let mut c = Correlator::new(1000);
        c.register(1, 100);
        c.register(2, 5000);
        // At now=2000, id 1 (sent 100, > timeout 1000 ago) is stale; id 2 (sent 5000) is not.
        let stale = c.sweep_timeouts(2000);
        assert_eq!(stale, vec![1]);
    }

    #[test]
    fn sweep_within_timeout_keeps_inflight() {
        let mut c = Correlator::new(1000);
        c.register(1, 100);
        // At now=500, id 1 (sent 100, only 400 ago) is fresh.
        let stale = c.sweep_timeouts(500);
        assert!(stale.is_empty());
        assert_eq!(c.pending_count(), 1);
    }

    #[test]
    fn many_inflight_pending_count() {
        let mut c = Correlator::new(1000);
        for i in 1..=50 {
            c.register(i, i * 10);
        }
        assert_eq!(c.pending_count(), 50);
    }

    #[test]
    fn sweep_clears_swept_ids() {
        let mut c = Correlator::new(1000);
        c.register(1, 100);
        c.sweep_timeouts(5000);
        assert_eq!(c.pending_count(), 0);
    }
}
