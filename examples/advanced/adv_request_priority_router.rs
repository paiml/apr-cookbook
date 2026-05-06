//! # Advanced Request Priority Router
//!
//! Multi-tenant inference: route requests to different GPU pools by:
//! - token count: long contexts → high-mem GPU pool
//! - tenant tier: paid users → premium pool
//! - latency-sensitivity: streaming UI → low-latency pool
//!
//! Routing rules (in order):
//! 1. Premium tier always to premium pool
//! 2. Streaming + small token count → low-latency pool
//! 3. Token count > 8k → high-memory pool
//! 4. Otherwise → general pool
//!
//! Demonstrates the **ADV.11** recipe for PMAT-141 (advanced round 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: AWS SageMaker multi-model endpoint routing.
//!
//! Run with: cargo run --example adv_request_priority_router
//!
//! Added by PMAT-141 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tenant {
    Free,
    Standard,
    Premium,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyClass {
    Streaming,
    Batch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pool {
    General,
    LowLatency,
    HighMemory,
    Premium,
}

#[derive(Debug, PartialEq)]
pub enum RouteVerdict {
    Ok { pool: Pool, priority: u8 },
    InvalidTokenCount,
}

const HIGH_MEMORY_THRESHOLD_TOKENS: u32 = 8_192;
const SMALL_TOKEN_THRESHOLD: u32 = 1_024;

pub fn route(tenant: Tenant, latency: LatencyClass, token_count: u32) -> RouteVerdict {
    if token_count == 0 {
        return RouteVerdict::InvalidTokenCount;
    }
    let (pool, priority) = if tenant == Tenant::Premium {
        (Pool::Premium, 0)
    } else if latency == LatencyClass::Streaming && token_count < SMALL_TOKEN_THRESHOLD {
        (Pool::LowLatency, 1)
    } else if token_count > HIGH_MEMORY_THRESHOLD_TOKENS {
        (Pool::HighMemory, 2)
    } else {
        (Pool::General, 3)
    };
    RouteVerdict::Ok { pool, priority }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_request_priority_router")?;

    println!(
        "premium streaming: {:?}",
        route(Tenant::Premium, LatencyClass::Streaming, 100)
    );
    println!(
        "free streaming small: {:?}",
        route(Tenant::Free, LatencyClass::Streaming, 100)
    );
    println!(
        "free batch large: {:?}",
        route(Tenant::Free, LatencyClass::Batch, 16_000)
    );
    println!(
        "standard normal: {:?}",
        route(Tenant::Standard, LatencyClass::Batch, 2_000)
    );
    println!("invalid: {:?}", route(Tenant::Free, LatencyClass::Batch, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn premium_always_premium_pool() {
        for token in [10u32, 1_000, 10_000] {
            for latency in [LatencyClass::Streaming, LatencyClass::Batch] {
                let v = route(Tenant::Premium, latency, token);
                if let RouteVerdict::Ok { pool, .. } = v {
                    assert_eq!(pool, Pool::Premium, "tokens={token} latency={latency:?}");
                }
            }
        }
    }

    #[test]
    fn streaming_small_low_latency() {
        let v = route(Tenant::Standard, LatencyClass::Streaming, 100);
        if let RouteVerdict::Ok { pool, .. } = v {
            assert_eq!(pool, Pool::LowLatency);
        }
    }

    #[test]
    fn large_context_high_memory() {
        let v = route(Tenant::Free, LatencyClass::Batch, 16_000);
        if let RouteVerdict::Ok { pool, .. } = v {
            assert_eq!(pool, Pool::HighMemory);
        }
    }

    #[test]
    fn medium_general() {
        let v = route(Tenant::Standard, LatencyClass::Batch, 2_000);
        if let RouteVerdict::Ok { pool, .. } = v {
            assert_eq!(pool, Pool::General);
        }
    }

    #[test]
    fn premium_priority_zero() {
        if let RouteVerdict::Ok { priority, .. } =
            route(Tenant::Premium, LatencyClass::Batch, 1_000)
        {
            assert_eq!(priority, 0);
        }
    }

    #[test]
    fn priority_order_matches_tier() {
        let prem = route(Tenant::Premium, LatencyClass::Streaming, 100);
        let lowlat = route(Tenant::Free, LatencyClass::Streaming, 100);
        let highmem = route(Tenant::Free, LatencyClass::Batch, 16_000);
        let gen = route(Tenant::Free, LatencyClass::Batch, 2_000);
        if let (
            RouteVerdict::Ok { priority: p1, .. },
            RouteVerdict::Ok { priority: p2, .. },
            RouteVerdict::Ok { priority: p3, .. },
            RouteVerdict::Ok { priority: p4, .. },
        ) = (prem, lowlat, highmem, gen)
        {
            assert!(p1 < p2);
            assert!(p2 < p3);
            assert!(p3 < p4);
        }
    }

    #[test]
    fn streaming_large_falls_to_high_mem() {
        // Streaming but tokens > small threshold and > high mem threshold.
        let v = route(Tenant::Free, LatencyClass::Streaming, 16_000);
        if let RouteVerdict::Ok { pool, .. } = v {
            assert_eq!(pool, Pool::HighMemory);
        }
    }

    #[test]
    fn invalid_zero_tokens() {
        assert_eq!(
            route(Tenant::Free, LatencyClass::Batch, 0),
            RouteVerdict::InvalidTokenCount
        );
    }

    #[test]
    fn boundary_at_small_threshold() {
        // exactly SMALL_TOKEN_THRESHOLD → no longer "small".
        let v = route(Tenant::Free, LatencyClass::Streaming, SMALL_TOKEN_THRESHOLD);
        if let RouteVerdict::Ok { pool, .. } = v {
            assert_eq!(pool, Pool::General);
        }
    }

    #[test]
    fn boundary_at_high_mem_threshold() {
        // exactly HIGH_MEMORY_THRESHOLD_TOKENS → not yet high-mem (strict >).
        let v = route(
            Tenant::Free,
            LatencyClass::Batch,
            HIGH_MEMORY_THRESHOLD_TOKENS,
        );
        if let RouteVerdict::Ok { pool, .. } = v {
            assert_eq!(pool, Pool::General);
        }
    }
}
