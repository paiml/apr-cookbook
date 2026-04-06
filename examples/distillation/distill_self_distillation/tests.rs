#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
use super::*;

fn test_model() -> SelfDistillModel {
    SelfDistillModel {
        name: "test-model".to_string(),
        num_layers: 8,
        hidden_size: 512,
        num_classes: 10,
        params_millions: 25.0,
    }
}

fn test_config() -> SelfDistillConfig {
    SelfDistillConfig {
        temperature: 3.0,
        alpha_kl: 0.6,
        alpha_aux: 0.2,
        alpha_task: 0.2,
        epochs: 10,
    }
}

#[test]
fn test_layer_representations() {
    let model = test_model();
    let reps = compute_layer_representations(&model).unwrap();

    assert_eq!(reps.len(), model.num_layers as usize);
    // First layer should be "input"
    assert_eq!(reps[0].role, "input");
    // Last layer should be "output"
    assert_eq!(reps[reps.len() - 1].role, "output");
}

#[test]
fn test_deeper_layers_higher_confidence() {
    let model = test_model();
    let reps = compute_layer_representations(&model).unwrap();

    // On average, deeper layers should have higher confidence
    let shallow_avg = reps[..3].iter().map(|r| r.confidence).sum::<f64>() / 3.0;
    let deep_avg = reps[5..].iter().map(|r| r.confidence).sum::<f64>() / 3.0;

    assert!(deep_avg > shallow_avg);
}

#[test]
fn test_distillation_pairs() {
    let model = test_model();
    let pairs = build_distillation_pairs(&model).unwrap();

    // Should have num_layers/2 pairs
    assert_eq!(pairs.len(), (model.num_layers / 2) as usize);

    // Teacher layers should be deeper than student layers
    for pair in &pairs {
        assert!(pair.teacher_layer > pair.student_layer);
    }
}

#[test]
fn test_self_distill_epoch_loss_decreases() {
    let config = test_config();

    let early = simulate_self_distill_epoch(1, &config).unwrap();
    let late = simulate_self_distill_epoch(10, &config).unwrap();

    assert!(late.total_loss < early.total_loss);
    assert!(late.kl_loss < early.kl_loss);
}

#[test]
fn test_self_distill_accuracy_improves() {
    let config = test_config();

    let early = simulate_self_distill_epoch(1, &config).unwrap();
    let late = simulate_self_distill_epoch(10, &config).unwrap();

    assert!(late.accuracy > early.accuracy);
}

#[test]
fn test_born_again_improves() {
    let config = test_config();
    let generations = simulate_born_again_generations(5, &config).unwrap();

    assert_eq!(generations.len(), 5);

    // Each generation should be at least as good as the previous
    for window in generations.windows(2) {
        assert!(window[1].accuracy >= window[0].accuracy);
    }
}

#[test]
fn test_born_again_diminishing_returns() {
    let config = test_config();
    let generations = simulate_born_again_generations(5, &config).unwrap();

    // First improvement should be larger than later improvements
    let first_gain = generations[1].accuracy - generations[0].accuracy;
    let last_gain = generations[4].accuracy - generations[3].accuracy;

    assert!(first_gain > last_gain);
}

#[test]
fn test_layer_accuracies_shallow_gain_more() {
    let model = test_model();
    let config = test_config();
    let accuracies = compute_layer_accuracies(&model, &config).unwrap();

    assert_eq!(accuracies.len(), model.num_layers as usize);

    // Shallow layers should gain more from self-distillation
    let shallow_gain = accuracies[0].after_sd - accuracies[0].before_sd;
    let deep_gain =
        accuracies[accuracies.len() - 1].after_sd - accuracies[accuracies.len() - 1].before_sd;

    assert!(shallow_gain > deep_gain);
}

#[test]
fn test_deterministic_epoch() {
    let config = test_config();

    let r1 = simulate_self_distill_epoch(5, &config).unwrap();
    let r2 = simulate_self_distill_epoch(5, &config).unwrap();

    assert_eq!(r1.kl_loss, r2.kl_loss);
    assert_eq!(r1.accuracy, r2.accuracy);
    assert_eq!(r1.total_loss, r2.total_loss);
}

#[test]
fn test_save_log() {
    let ctx = RecipeContext::new("test_sd_save_log").unwrap();
    let path = ctx.path("log.json");

    let log = vec![EpochResult {
        epoch: 1,
        kl_loss: 1.0,
        aux_loss: 0.5,
        task_loss: 0.3,
        total_loss: 0.7,
        accuracy: 0.8,
    }];

    save_log(&path, &log).unwrap();
    assert!(path.exists());
}

#[test]
fn test_save_generations() {
    let ctx = RecipeContext::new("test_sd_save_gen").unwrap();
    let path = ctx.path("gen.json");

    let generations = vec![GenerationResult {
        generation: 0,
        accuracy: 0.9,
        final_loss: 0.3,
    }];

    save_generations(&path, &generations).unwrap();
    assert!(path.exists());
}

#[test]
fn test_weighted_loss_components() {
    let config = test_config();
    let result = simulate_self_distill_epoch(5, &config).unwrap();

    // Total loss should be the weighted sum
    let expected_total = config.alpha_kl * result.kl_loss
        + config.alpha_aux * result.aux_loss
        + config.alpha_task * result.task_loss;

    assert!((result.total_loss - expected_total).abs() < 1e-10);
}
