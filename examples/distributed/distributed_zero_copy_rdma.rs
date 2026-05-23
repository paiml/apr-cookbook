//! # Distributed Zero-Copy RDMA Picker
//!
//! RDMA (Remote Direct Memory Access) skips the kernel/TCP stack but
//! has fixed setup cost. Picker rule:
//!   transfer_size_bytes < 8 KiB     → TcpAvoidRdma (RDMA setup dominates)
//!   8 KiB ≤ size < 1 MiB           → RdmaInline (small messages, ~µs)
//!   size ≥ 1 MiB                    → RdmaWithVerbs (large bulk transfer)
//!
//! Returns expected_throughput_gbps + tier.
//!
//! Demonstrates the **DIST.10** recipe for PMAT-142 (distributed coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NVIDIA RDMA over Converged Ethernet (RoCE) latency table.
//!
//! Run with: cargo run --example distributed_zero_copy_rdma
//!
//! Added by PMAT-142 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transport {
    TcpAvoidRdma,
    RdmaInline,
    RdmaWithVerbs,
}

#[derive(Debug, PartialEq)]
pub enum TransportVerdict {
    Ok {
        transport: Transport,
        expected_throughput_gbps: f64,
    },
    InvalidSize,
}

const TCP_THRESHOLD_BYTES: u64 = 8 * 1024;
const VERBS_THRESHOLD_BYTES: u64 = 1024 * 1024;

pub fn pick(transfer_size_bytes: u64, link_bandwidth_gbps: f64) -> TransportVerdict {
    if transfer_size_bytes == 0 || !link_bandwidth_gbps.is_finite() || link_bandwidth_gbps <= 0.0 {
        return TransportVerdict::InvalidSize;
    }
    let transport = if transfer_size_bytes < TCP_THRESHOLD_BYTES {
        Transport::TcpAvoidRdma
    } else if transfer_size_bytes < VERBS_THRESHOLD_BYTES {
        Transport::RdmaInline
    } else {
        Transport::RdmaWithVerbs
    };
    let efficiency = match transport {
        Transport::TcpAvoidRdma => 0.4,
        Transport::RdmaInline => 0.7,
        Transport::RdmaWithVerbs => 0.95,
    };
    let expected_throughput_gbps = link_bandwidth_gbps * efficiency;
    TransportVerdict::Ok {
        transport,
        expected_throughput_gbps,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distributed_zero_copy_rdma")?;

    println!("4 KiB on 100 Gbps: {:?}", pick(4 * 1024, 100.0));
    println!("64 KiB on 100 Gbps: {:?}", pick(64 * 1024, 100.0));
    println!("4 MiB on 100 Gbps: {:?}", pick(4 * 1024 * 1024, 100.0));
    println!("invalid 0: {:?}", pick(0, 100.0));
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
    fn small_picks_tcp() {
        let v = pick(4 * 1024, 100.0);
        if let TransportVerdict::Ok { transport, .. } = v {
            assert_eq!(transport, Transport::TcpAvoidRdma);
        }
    }

    #[test]
    fn medium_picks_rdma_inline() {
        let v = pick(64 * 1024, 100.0);
        if let TransportVerdict::Ok { transport, .. } = v {
            assert_eq!(transport, Transport::RdmaInline);
        }
    }

    #[test]
    fn large_picks_verbs() {
        let v = pick(4 * 1024 * 1024, 100.0);
        if let TransportVerdict::Ok { transport, .. } = v {
            assert_eq!(transport, Transport::RdmaWithVerbs);
        }
    }

    #[test]
    fn zero_size_invalid() {
        assert_eq!(pick(0, 100.0), TransportVerdict::InvalidSize);
    }

    #[test]
    fn zero_bandwidth_invalid() {
        assert_eq!(pick(4096, 0.0), TransportVerdict::InvalidSize);
    }

    #[test]
    fn nan_bandwidth_invalid() {
        assert_eq!(pick(4096, f64::NAN), TransportVerdict::InvalidSize);
    }

    #[test]
    fn verbs_efficiency_highest() {
        let tcp = pick(4 * 1024, 100.0);
        let inline = pick(64 * 1024, 100.0);
        let verbs = pick(4 * 1024 * 1024, 100.0);
        if let (
            TransportVerdict::Ok {
                expected_throughput_gbps: t,
                ..
            },
            TransportVerdict::Ok {
                expected_throughput_gbps: i,
                ..
            },
            TransportVerdict::Ok {
                expected_throughput_gbps: v,
                ..
            },
        ) = (tcp, inline, verbs)
        {
            assert!(v > i);
            assert!(i > t);
        }
    }

    #[test]
    fn boundary_at_8kib_picks_rdma_inline() {
        // exactly 8 KiB = TCP_THRESHOLD → falls into rdma_inline.
        let v = pick(TCP_THRESHOLD_BYTES, 100.0);
        if let TransportVerdict::Ok { transport, .. } = v {
            assert_eq!(transport, Transport::RdmaInline);
        }
    }

    #[test]
    fn boundary_at_1mib_picks_verbs() {
        let v = pick(VERBS_THRESHOLD_BYTES, 100.0);
        if let TransportVerdict::Ok { transport, .. } = v {
            assert_eq!(transport, Transport::RdmaWithVerbs);
        }
    }

    #[test]
    fn throughput_proportional_to_link_speed() {
        let v_50 = pick(4 * 1024 * 1024, 50.0);
        let v_100 = pick(4 * 1024 * 1024, 100.0);
        if let (
            TransportVerdict::Ok {
                expected_throughput_gbps: t50,
                ..
            },
            TransportVerdict::Ok {
                expected_throughput_gbps: t100,
                ..
            },
        ) = (v_50, v_100)
        {
            assert!((t100 / t50 - 2.0).abs() < 1e-9);
        }
    }
}
