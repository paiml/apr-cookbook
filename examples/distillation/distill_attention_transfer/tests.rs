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

    fn teacher_spec() -> AttentionModelSpec {
        AttentionModelSpec {
            name: "teacher".to_string(),
            layers: 12,
            hidden_size: 768,
            num_heads: 12,
            head_dim: 64,
            seq_len: 128,
            params_millions: 110.0,
        }
    }

    fn student_spec() -> AttentionModelSpec {
        AttentionModelSpec {
            name: "student".to_string(),
            layers: 4,
            hidden_size: 256,
            num_heads: 4,
            head_dim: 64,
            seq_len: 128,
            params_millions: 6.5,
        }
    }

    fn default_config() -> AttentionTransferConfig {
        AttentionTransferConfig {
            epochs: 10,
            learning_rate: 1e-4,
            beta: 0.5,
        }
    }

    #[test]
    fn test_build_layer_mappings() {
        let teacher = teacher_spec();
        let student = student_spec();
        let mappings = build_layer_mappings(&teacher, &student).unwrap();

        assert_eq!(mappings.len(), 4);
        assert_eq!(mappings[0].teacher_layer, 0);
        assert_eq!(mappings[3].teacher_layer, 11);
    }

    #[test]
    fn test_build_layer_mappings_heads() {
        let teacher = teacher_spec();
        let student = student_spec();
        let mappings = build_layer_mappings(&teacher, &student).unwrap();

        for mapping in &mappings {
            assert_eq!(mapping.teacher_heads, 12);
            assert_eq!(mapping.student_heads, 4);
        }
    }

    #[test]
    fn test_build_layer_mappings_zero_student_layers() {
        let teacher = teacher_spec();
        let mut student = student_spec();
        student.layers = 0;

        let result = build_layer_mappings(&teacher, &student);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_projection_linear() {
        let mapping = AttentionLayerMapping {
            name: "early".to_string(),
            teacher_layer: 0,
            student_layer: 0,
            teacher_heads: 12,
            student_heads: 4,
        };

        let proj = compute_projection(&mapping).unwrap();

        assert_eq!(proj.projection_type, "Linear");
        assert_eq!(proj.projection_params, 48); // 4 * 12
    }

    #[test]
    fn test_compute_projection_identity() {
        let mapping = AttentionLayerMapping {
            name: "same".to_string(),
            teacher_layer: 0,
            student_layer: 0,
            teacher_heads: 8,
            student_heads: 8,
        };

        let proj = compute_projection(&mapping).unwrap();

        assert_eq!(proj.projection_type, "Identity");
        assert_eq!(proj.projection_params, 0);
    }

    #[test]
    fn test_epoch_result_teacher_constant() {
        let config = default_config();
        let projections = vec![AttentionProjection {
            layer_name: "test".to_string(),
            teacher_attn_shape: (12, 128),
            student_attn_shape: (4, 128),
            projection_type: "Linear".to_string(),
            projection_params: 48,
        }];

        let r1 = simulate_attention_transfer_epoch(1, &config, &projections).unwrap();
        let r5 = simulate_attention_transfer_epoch(5, &config, &projections).unwrap();

        assert_eq!(r1.teacher_accuracy, r5.teacher_accuracy);
    }

    #[test]
    fn test_student_improves_over_epochs() {
        let config = default_config();
        let projections = vec![AttentionProjection {
            layer_name: "test".to_string(),
            teacher_attn_shape: (12, 128),
            student_attn_shape: (4, 128),
            projection_type: "Linear".to_string(),
            projection_params: 48,
        }];

        let early = simulate_attention_transfer_epoch(1, &config, &projections).unwrap();
        let late = simulate_attention_transfer_epoch(10, &config, &projections).unwrap();

        assert!(late.student_accuracy > early.student_accuracy);
    }

    #[test]
    fn test_attention_loss_decreases() {
        let config = default_config();
        let projections = vec![AttentionProjection {
            layer_name: "test".to_string(),
            teacher_attn_shape: (12, 128),
            student_attn_shape: (4, 128),
            projection_type: "Linear".to_string(),
            projection_params: 48,
        }];

        let early = simulate_attention_transfer_epoch(1, &config, &projections).unwrap();
        let late = simulate_attention_transfer_epoch(10, &config, &projections).unwrap();

        assert!(late.attention_transfer_loss < early.attention_transfer_loss);
    }

    #[test]
    fn test_total_loss_combines_components() {
        let config = AttentionTransferConfig {
            epochs: 10,
            learning_rate: 1e-4,
            beta: 0.5,
        };
        let projections = vec![AttentionProjection {
            layer_name: "test".to_string(),
            teacher_attn_shape: (12, 128),
            student_attn_shape: (4, 128),
            projection_type: "Linear".to_string(),
            projection_params: 48,
        }];

        let result = simulate_attention_transfer_epoch(5, &config, &projections).unwrap();

        let expected_total =
            (1.0 - config.beta) * result.task_loss + config.beta * result.attention_transfer_loss;
        assert!((result.total_loss - expected_total).abs() < 1e-10);
    }

    #[test]
    fn test_layer_attention_quality_bounded() {
        let proj = AttentionProjection {
            layer_name: "test_layer".to_string(),
            teacher_attn_shape: (12, 128),
            student_attn_shape: (4, 128),
            projection_type: "Linear".to_string(),
            projection_params: 48,
        };

        let quality = compute_layer_attention_quality(&proj).unwrap();

        assert!(quality.attention_mse > 0.0);
        assert!(quality.cosine_similarity >= 0.0);
        assert!(quality.cosine_similarity <= 1.0);
        assert!(!quality.quality_label.is_empty());
    }

    #[test]
    fn test_deterministic_epoch() {
        let config = default_config();
        let projections = vec![AttentionProjection {
            layer_name: "test".to_string(),
            teacher_attn_shape: (12, 128),
            student_attn_shape: (4, 128),
            projection_type: "Linear".to_string(),
            projection_params: 48,
        }];

        let r1 = simulate_attention_transfer_epoch(5, &config, &projections).unwrap();
        let r2 = simulate_attention_transfer_epoch(5, &config, &projections).unwrap();

        assert_eq!(r1.student_accuracy, r2.student_accuracy);
        assert_eq!(r1.attention_transfer_loss, r2.attention_transfer_loss);
        assert_eq!(r1.total_loss, r2.total_loss);
    }

    #[test]
    fn test_save_log() {
        let ctx = RecipeContext::new("test_attn_save").unwrap();
        let path = ctx.path("log.json");

        let log = vec![EpochResult {
            epoch: 1,
            task_loss: 1.5,
            attention_transfer_loss: 0.8,
            total_loss: 1.15,
            teacher_accuracy: 0.92,
            student_accuracy: 0.45,
        }];

        save_log(&path, &log).unwrap();
        assert!(path.exists());
    }
}

#[cfg(test)]
mod proptests {
    use super::super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_student_accuracy_bounded(epoch in 1u32..100) {
            let config = AttentionTransferConfig {
                epochs: 100,
                learning_rate: 1e-4,
                beta: 0.5,
            };
            let projections = vec![AttentionProjection {
                layer_name: "test".to_string(),
                teacher_attn_shape: (12, 128),
                student_attn_shape: (4, 128),
                projection_type: "Linear".to_string(),
                projection_params: 48,
            }];

            let result = simulate_attention_transfer_epoch(epoch, &config, &projections).unwrap();

            prop_assert!(result.student_accuracy >= 0.0);
            prop_assert!(result.student_accuracy <= result.teacher_accuracy);
        }

        #[test]
        fn prop_attention_loss_positive(epoch in 1u32..50) {
            let config = AttentionTransferConfig {
                epochs: 50,
                learning_rate: 1e-4,
                beta: 0.5,
            };
            let projections = vec![AttentionProjection {
                layer_name: "test".to_string(),
                teacher_attn_shape: (12, 128),
                student_attn_shape: (4, 128),
                projection_type: "Linear".to_string(),
                projection_params: 48,
            }];

            let result = simulate_attention_transfer_epoch(epoch, &config, &projections).unwrap();

            prop_assert!(result.attention_transfer_loss > 0.0);
            prop_assert!(result.task_loss > 0.0);
            prop_assert!(result.total_loss > 0.0);
        }

        #[test]
        fn prop_total_loss_is_weighted_sum(epoch in 1u32..50, beta in 0.0f64..1.0) {
            let config = AttentionTransferConfig {
                epochs: 50,
                learning_rate: 1e-4,
                beta,
            };
            let projections = vec![AttentionProjection {
                layer_name: "test".to_string(),
                teacher_attn_shape: (12, 128),
                student_attn_shape: (4, 128),
                projection_type: "Linear".to_string(),
                projection_params: 48,
            }];

            let result = simulate_attention_transfer_epoch(epoch, &config, &projections).unwrap();

            let expected = (1.0 - beta) * result.task_loss + beta * result.attention_transfer_loss;
            prop_assert!((result.total_loss - expected).abs() < 1e-10);
        }

        #[test]
        fn prop_layer_quality_cosine_bounded(name in "[a-z]{3,10}") {
            let proj = AttentionProjection {
                layer_name: name,
                teacher_attn_shape: (12, 128),
                student_attn_shape: (4, 128),
                projection_type: "Linear".to_string(),
                projection_params: 48,
            };

            let quality = compute_layer_attention_quality(&proj).unwrap();

            prop_assert!(quality.cosine_similarity >= 0.0);
            prop_assert!(quality.cosine_similarity <= 1.0);
            prop_assert!(quality.attention_mse > 0.0);
        }
    }
}
