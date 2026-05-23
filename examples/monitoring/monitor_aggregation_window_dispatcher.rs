//! # Monitoring Aggregation Window Dispatcher
//!
//! Stream aggregations use windows: tumbling (disjoint, fixed-size),
//! sliding (overlapping, every Δt), hopping (slide by stride < length),
//! session (gap-based, no fixed length). This recipe builds the
//! window-type dispatcher + bucket-count calculator over a stream.
//!
//! Demonstrates the **MON.10** recipe for PMAT-124 (monitoring coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Carbone et al. (2017). Stream Processing with Apache Flink (chap. 4).
//!
//! Run with: cargo run --example monitor_aggregation_window_dispatcher
//!
//! Added by PMAT-124 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowKind {
    Tumbling,
    Sliding,
    Hopping,
    Session,
}

#[derive(Debug, PartialEq)]
pub enum DispatchVerdict {
    Ok { kind: WindowKind, num_windows: u32 },
    InvalidWindowSize,
    InvalidStride,
    StrideExceedsWindow,
    InvalidStream,
}

pub fn dispatch(
    stream_seconds: u64,
    window_seconds: u64,
    stride_seconds: u64,
    session_gap_seconds: Option<u64>,
) -> DispatchVerdict {
    if stream_seconds == 0 {
        return DispatchVerdict::InvalidStream;
    }
    if let Some(_gap) = session_gap_seconds {
        // Session windows: caller decides bucket count externally.
        return DispatchVerdict::Ok {
            kind: WindowKind::Session,
            num_windows: 0,
        };
    }
    if window_seconds == 0 {
        return DispatchVerdict::InvalidWindowSize;
    }
    if stride_seconds == 0 {
        return DispatchVerdict::InvalidStride;
    }
    if stride_seconds > window_seconds {
        return DispatchVerdict::StrideExceedsWindow;
    }
    let kind = if stride_seconds == window_seconds {
        WindowKind::Tumbling
    } else if stride_seconds == 1 || stride_seconds * 4 < window_seconds {
        WindowKind::Sliding
    } else {
        WindowKind::Hopping
    };
    let num_windows = stream_seconds.div_ceil(stride_seconds);
    DispatchVerdict::Ok {
        kind,
        num_windows: num_windows as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_aggregation_window_dispatcher")?;

    let cases = [
        (3600u64, 60u64, 60u64, None),
        (3600, 60, 1, None),
        (3600, 60, 20, None),
        (3600, 60, 90, None),
        (3600, 0, 60, None),
        (0, 60, 60, None),
        (3600, 60, 60, Some(30)),
    ];
    for (stream, win, stride, gap) in cases {
        println!(
            "stream={stream}s win={win}s stride={stride}s gap={gap:?}  →  {:?}",
            dispatch(stream, win, stride, gap)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn equal_window_and_stride_tumbling() {
        let v = dispatch(3600, 60, 60, None);
        assert!(matches!(
            v,
            DispatchVerdict::Ok {
                kind: WindowKind::Tumbling,
                ..
            }
        ));
    }

    #[test]
    fn unit_stride_classified_sliding() {
        let v = dispatch(3600, 60, 1, None);
        assert!(matches!(
            v,
            DispatchVerdict::Ok {
                kind: WindowKind::Sliding,
                ..
            }
        ));
    }

    #[test]
    fn small_stride_relative_to_window_sliding() {
        // stride 10 < window 60 / 4 → sliding.
        let v = dispatch(3600, 60, 10, None);
        assert!(matches!(
            v,
            DispatchVerdict::Ok {
                kind: WindowKind::Sliding,
                ..
            }
        ));
    }

    #[test]
    fn medium_stride_classified_hopping() {
        // stride 30, window 60 → hopping.
        let v = dispatch(3600, 60, 30, None);
        assert!(matches!(
            v,
            DispatchVerdict::Ok {
                kind: WindowKind::Hopping,
                ..
            }
        ));
    }

    #[test]
    fn session_gap_yields_session_kind() {
        let v = dispatch(3600, 60, 60, Some(30));
        assert!(matches!(
            v,
            DispatchVerdict::Ok {
                kind: WindowKind::Session,
                ..
            }
        ));
    }

    #[test]
    fn stride_exceeds_window_rejected() {
        assert_eq!(
            dispatch(3600, 60, 90, None),
            DispatchVerdict::StrideExceedsWindow
        );
    }

    #[test]
    fn zero_window_rejected() {
        assert_eq!(
            dispatch(3600, 0, 60, None),
            DispatchVerdict::InvalidWindowSize
        );
    }

    #[test]
    fn zero_stride_rejected() {
        assert_eq!(dispatch(3600, 60, 0, None), DispatchVerdict::InvalidStride);
    }

    #[test]
    fn zero_stream_rejected() {
        assert_eq!(dispatch(0, 60, 60, None), DispatchVerdict::InvalidStream);
    }

    #[test]
    fn tumbling_window_count_correct() {
        // 3600 sec / 60 sec = 60 windows.
        if let DispatchVerdict::Ok { num_windows, .. } = dispatch(3600, 60, 60, None) {
            assert_eq!(num_windows, 60);
        }
    }

    #[test]
    fn sliding_window_count_more_than_tumbling() {
        let tumb = dispatch(3600, 60, 60, None);
        let slide = dispatch(3600, 60, 1, None);
        if let (
            DispatchVerdict::Ok { num_windows: t, .. },
            DispatchVerdict::Ok { num_windows: s, .. },
        ) = (tumb, slide)
        {
            assert!(s > t);
        }
    }
}
