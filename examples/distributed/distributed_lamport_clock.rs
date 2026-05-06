//! # Distributed Lamport Clock
//!
//! Lamport clocks order events across nodes:
//!   on local event: clock += 1
//!   on send: include clock in message; clock += 1
//!   on receive: clock = max(clock, msg_clock) + 1
//!
//! Comparison: events a < b iff (a.clock < b.clock) or (a.clock == b.clock
//! AND a.node < b.node). This recipe builds the clock + comparator.
//!
//! Demonstrates the **DIST.8** recipe for PMAT-139 (distributed coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Lamport (1978). Time, Clocks, and the Ordering of Events.
//!
//! Run with: cargo run --example distributed_lamport_clock
//!
//! Added by PMAT-139 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LamportEvent {
    pub clock: u64,
    pub node_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    Before,
    After,
    Concurrent,
}

pub struct LamportClock {
    counter: u64,
    node_id: u32,
}

impl LamportClock {
    pub fn new(node_id: u32) -> Self {
        Self {
            counter: 0,
            node_id,
        }
    }

    pub fn local_event(&mut self) -> LamportEvent {
        self.counter += 1;
        LamportEvent {
            clock: self.counter,
            node_id: self.node_id,
        }
    }

    pub fn send(&mut self) -> LamportEvent {
        self.local_event()
    }

    pub fn receive(&mut self, incoming: LamportEvent) -> LamportEvent {
        self.counter = self.counter.max(incoming.clock) + 1;
        LamportEvent {
            clock: self.counter,
            node_id: self.node_id,
        }
    }

    pub fn current_value(&self) -> u64 {
        self.counter
    }
}

pub fn compare(a: LamportEvent, b: LamportEvent) -> Order {
    if a.clock < b.clock {
        Order::Before
    } else if a.clock > b.clock {
        Order::After
    } else if a.node_id == b.node_id {
        Order::Before
    } else {
        Order::Concurrent
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distributed_lamport_clock")?;

    let mut node_a = LamportClock::new(1);
    let mut node_b = LamportClock::new(2);

    let a1 = node_a.local_event();
    let a2 = node_a.send();
    let b1 = node_b.receive(a2);
    let b2 = node_b.local_event();

    println!("a1: {a1:?}");
    println!("a2: {a2:?}");
    println!("b1: {b1:?}");
    println!("b2: {b2:?}");
    println!("a1 vs b1: {:?}", compare(a1, b1));
    println!("a2 vs b1: {:?}", compare(a2, b1));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn local_event_increments_counter() {
        let mut c = LamportClock::new(1);
        let e = c.local_event();
        assert_eq!(e.clock, 1);
        let e2 = c.local_event();
        assert_eq!(e2.clock, 2);
    }

    #[test]
    fn receive_updates_max_plus_one() {
        let mut c = LamportClock::new(1);
        c.local_event(); // clock=1
        let incoming = LamportEvent {
            clock: 5,
            node_id: 99,
        };
        let after = c.receive(incoming);
        assert_eq!(after.clock, 6);
    }

    #[test]
    fn receive_below_local_still_increments() {
        let mut c = LamportClock::new(1);
        c.local_event();
        c.local_event();
        c.local_event(); // clock=3
        let incoming = LamportEvent {
            clock: 1,
            node_id: 99,
        };
        let after = c.receive(incoming);
        assert_eq!(after.clock, 4);
    }

    #[test]
    fn compare_lower_clock_before() {
        let a = LamportEvent {
            clock: 1,
            node_id: 1,
        };
        let b = LamportEvent {
            clock: 2,
            node_id: 1,
        };
        assert_eq!(compare(a, b), Order::Before);
    }

    #[test]
    fn compare_higher_clock_after() {
        let a = LamportEvent {
            clock: 5,
            node_id: 1,
        };
        let b = LamportEvent {
            clock: 2,
            node_id: 1,
        };
        assert_eq!(compare(a, b), Order::After);
    }

    #[test]
    fn compare_equal_clock_different_nodes_concurrent() {
        let a = LamportEvent {
            clock: 3,
            node_id: 1,
        };
        let b = LamportEvent {
            clock: 3,
            node_id: 2,
        };
        assert_eq!(compare(a, b), Order::Concurrent);
    }

    #[test]
    fn compare_equal_clock_same_node_before() {
        let a = LamportEvent {
            clock: 3,
            node_id: 1,
        };
        let b = LamportEvent {
            clock: 3,
            node_id: 1,
        };
        assert_eq!(compare(a, b), Order::Before);
    }

    #[test]
    fn send_increments_like_local() {
        let mut c = LamportClock::new(1);
        let e1 = c.send();
        let e2 = c.send();
        assert_eq!(e1.clock, 1);
        assert_eq!(e2.clock, 2);
    }

    #[test]
    fn current_value_tracks_counter() {
        let mut c = LamportClock::new(1);
        c.local_event();
        c.local_event();
        c.local_event();
        assert_eq!(c.current_value(), 3);
    }

    #[test]
    fn happens_before_chain_preserved() {
        // a → b → c chain transitively orders.
        let a = LamportEvent {
            clock: 1,
            node_id: 1,
        };
        let b = LamportEvent {
            clock: 2,
            node_id: 1,
        };
        let c = LamportEvent {
            clock: 3,
            node_id: 1,
        };
        assert_eq!(compare(a, b), Order::Before);
        assert_eq!(compare(b, c), Order::Before);
        assert_eq!(compare(a, c), Order::Before);
    }
}
