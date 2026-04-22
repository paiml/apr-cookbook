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

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test_parse_args_demo() {
        let args = vec!["apr-ptx-map".to_string(), "--demo".to_string()];
        let config = parse_args(&args).expect("parse ok");
        assert!(config.demo);
        assert!(config.kernel_filter.is_none());
    }

    #[test]
    fn test_parse_args_kernel_filter() {
        let args = vec![
            "apr-ptx-map".to_string(),
            "--kernel-filter".to_string(),
            "attention".to_string(),
        ];
        let config = parse_args(&args).expect("parse ok");
        assert_eq!(config.kernel_filter, Some("attention".to_string()));
    }

    #[test]
    fn test_parse_args_model_path() {
        let args = vec!["apr-ptx-map".to_string(), "model.apr".to_string()];
        let config = parse_args(&args).expect("parse ok");
        assert_eq!(config.model_path, Some("model.apr".to_string()));
    }

    #[test]
    fn test_parse_args_unknown_rejected() {
        let args = vec!["apr-ptx-map".to_string(), "--bogus".to_string()];
        assert!(parse_args(&args).is_err());
    }

    #[test]
    fn test_demo_mappings_count() {
        let mappings = create_demo_mappings();
        assert_eq!(mappings.len(), DEMO_LAYERS.len());
    }

    #[test]
    fn test_threads_per_block() {
        let m = KernelMapping {
            layer_name: "test".to_string(),
            kernel_name: "k_test".to_string(),
            grid_dim: [4, 2, 1],
            block_dim: [128, 2, 1],
            shared_mem_bytes: 0,
            registers_per_thread: 32,
        };
        assert_eq!(m.threads_per_block(), 256);
        assert_eq!(m.total_blocks(), 8);
    }

    #[test]
    fn test_occupancy_full() {
        // Low register count, no shared memory -> should yield high occupancy
        let m = KernelMapping {
            layer_name: "test".to_string(),
            kernel_name: "k_test".to_string(),
            grid_dim: [1, 1, 1],
            block_dim: [256, 1, 1],
            shared_mem_bytes: 0,
            registers_per_thread: 16,
        };
        let occ = compute_occupancy(&m);
        assert!(
            occ > 90.0,
            "low-resource kernel should have high occupancy, got {:.1}%",
            occ
        );
    }

    #[test]
    fn test_occupancy_register_limited() {
        // Very high register count -> reduced occupancy
        let m = KernelMapping {
            layer_name: "test".to_string(),
            kernel_name: "k_test".to_string(),
            grid_dim: [1, 1, 1],
            block_dim: [128, 1, 1],
            shared_mem_bytes: 0,
            registers_per_thread: 128,
        };
        let occ = compute_occupancy(&m);
        assert!(
            occ < 50.0,
            "high-register kernel should have reduced occupancy, got {:.1}%",
            occ
        );
    }

    #[test]
    fn test_occupancy_shmem_limited() {
        // Near-limit shared memory -> reduced occupancy
        let m = KernelMapping {
            layer_name: "test".to_string(),
            kernel_name: "k_test".to_string(),
            grid_dim: [1, 1, 1],
            block_dim: [128, 1, 1],
            shared_mem_bytes: 48 * 1024, // fills entire SM shmem budget
            registers_per_thread: 32,
        };
        let occ = compute_occupancy(&m);
        // Only 1 block fits per SM
        assert!(
            occ < 20.0,
            "shmem-limited kernel should have low occupancy, got {:.1}%",
            occ
        );
    }

    #[test]
    fn test_ptx_regions_cover_all_kernels() {
        let mappings = create_demo_mappings();
        let regions = generate_ptx_regions(&mappings);
        // Each mapping produces 3 regions (compute, memory, control)
        assert_eq!(regions.len(), mappings.len() * 3);
    }

    #[test]
    fn test_ptx_regions_categories() {
        let mappings = create_demo_mappings();
        let regions = generate_ptx_regions(&mappings);
        for chunk in regions.chunks(3) {
            assert_eq!(chunk[0].category, InstructionCategory::Compute);
            assert_eq!(chunk[1].category, InstructionCategory::Memory);
            assert_eq!(chunk[2].category, InstructionCategory::Control);
        }
    }

    #[test]
    fn test_ptx_regions_non_overlapping() {
        let mappings = create_demo_mappings();
        let regions = generate_ptx_regions(&mappings);
        for window in regions.windows(2) {
            assert!(
                window[1].start_line > window[0].end_line,
                "regions must not overlap: [{}-{}] vs [{}-{}]",
                window[0].start_line,
                window[0].end_line,
                window[1].start_line,
                window[1].end_line,
            );
        }
    }

    #[test]
    fn test_ptx_region_line_span() {
        let r = PtxSourceRegion {
            kernel_name: "test".to_string(),
            start_line: 10,
            end_line: 19,
            instruction_count: 10,
            category: InstructionCategory::Compute,
        };
        assert_eq!(r.line_span(), 10);
    }

    #[test]
    fn test_instruction_category_display() {
        assert_eq!(InstructionCategory::Compute.to_string(), "compute");
        assert_eq!(InstructionCategory::Memory.to_string(), "memory");
        assert_eq!(InstructionCategory::Control.to_string(), "control");
    }

    #[test]
    fn test_demo_run() {
        let config = PtxMapConfig {
            model_path: None,
            kernel_filter: None,
            demo: true,
        };
        assert!(run_ptx_map(&config).is_ok());
    }

    #[test]
    fn test_demo_run_with_filter() {
        let config = PtxMapConfig {
            model_path: None,
            kernel_filter: Some("attention".to_string()),
            demo: true,
        };
        assert!(run_ptx_map(&config).is_ok());
    }

    #[test]
    fn test_deterministic_seed_consistent() {
        let s1 = deterministic_seed("test_kernel");
        let s2 = deterministic_seed("test_kernel");
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_deterministic_seed_varies() {
        let s1 = deterministic_seed("kernel_a");
        let s2 = deterministic_seed("kernel_b");
        assert_ne!(s1, s2);
    }
}

#[cfg(test)]
mod proptests {
    use super::super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_occupancy_bounded(regs in 16u32..256, shmem_kb in 0u32..64, block_x in 1u32..9) {
            let block_size = block_x * 32; // warp-aligned
            let m = KernelMapping {
                layer_name: "prop".to_string(),
                kernel_name: "k_prop".to_string(),
                grid_dim: [1, 1, 1],
                block_dim: [block_size, 1, 1],
                shared_mem_bytes: shmem_kb * 1024,
                registers_per_thread: regs,
            };
            let occ = compute_occupancy(&m);
            prop_assert!(occ >= 0.0 && occ <= 100.0,
                "occupancy must be in [0,100], got {}", occ);
        }

        #[test]
        fn prop_ptx_regions_monotonic(n in 1usize..8) {
            let mappings: Vec<KernelMapping> = (0..n).map(|i| KernelMapping {
                layer_name: format!("layer_{}", i),
                kernel_name: format!("k_{}", i),
                grid_dim: [1, 1, 1],
                block_dim: [128, 1, 1],
                shared_mem_bytes: 4096,
                registers_per_thread: 32,
            }).collect();
            let regions = generate_ptx_regions(&mappings);
            for window in regions.windows(2) {
                prop_assert!(window[1].start_line > window[0].end_line,
                    "regions must be strictly ordered");
            }
        }
    }
}
