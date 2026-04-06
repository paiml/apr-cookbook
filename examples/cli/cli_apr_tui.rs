//! # Recipe: APR Model TUI (Headless Simulation)
//! **CLI Equivalent**: `apr tui`
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/cli-parity-v1.yaml
//!
//! Simulate a terminal UI for interactive model exploration, rendered in
//! headless mode. Mirrors `apr tui` with 4 tabs: Overview, Tensors, Stats, Help.
//!
//! ```bash
//! cargo run --example cli_apr_tui
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr inspect model.apr          # APR native format
//! apr inspect model.gguf         # GGUF (llama.cpp compatible)
//! apr inspect model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Amershi, S. et al. (2019). *Software Engineering for Machine Learning: A Case Study*. ICSE. DOI: 10.1109/ICSE-SEIP.2019.00042

use apr_cookbook::prelude::*;
use rand::Rng;

const NUM_TENSORS: usize = 24;
const PAGE_SIZE: usize = 10;
const BOX_TL: char = '+';
const BOX_TR: char = '+';
const BOX_BL: char = '+';
const BOX_BR: char = '+';
const BOX_H: char = '-';
const BOX_V: char = '|';

// ---- Domain Types -----------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Overview,
    Tensors,
    Stats,
    Help,
}

impl Tab {
    const ALL: [Tab; 4] = [Tab::Overview, Tab::Tensors, Tab::Stats, Tab::Help];
    fn label(self) -> &'static str {
        match self {
            Self::Overview => "Overview",
            Self::Tensors => "Tensors",
            Self::Stats => "Stats",
            Self::Help => "Help",
        }
    }
    fn index(self) -> usize {
        match self {
            Self::Overview => 1,
            Self::Tensors => 2,
            Self::Stats => 3,
            Self::Help => 4,
        }
    }
    #[allow(dead_code)]
    fn next(self) -> Self {
        match self {
            Self::Overview => Self::Tensors,
            Self::Tensors => Self::Stats,
            Self::Stats => Self::Help,
            Self::Help => Self::Overview,
        }
    }
}

#[derive(Debug, Clone)]
struct TensorInfo {
    name: String,
    shape: Vec<usize>,
    dtype: String,
    size_bytes: usize,
}
impl TensorInfo {
    fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Debug, Clone)]
struct AppState {
    current_tab: Tab,
    model_name: String,
    model_version: String,
    format_version: String,
    creation_date: String,
    tensors: Vec<TensorInfo>,
    page: usize,
    page_size: usize,
}

impl AppState {
    fn total_params(&self) -> usize {
        self.tensors.iter().map(TensorInfo::num_elements).sum()
    }
    fn total_size_bytes(&self) -> usize {
        self.tensors.iter().map(|t| t.size_bytes).sum()
    }
    fn compression_ratio(&self) -> f64 {
        let stored = self.total_size_bytes();
        if stored == 0 {
            return 1.0;
        }
        (self.total_params() * 4) as f64 / stored as f64
    }
    fn total_pages(&self) -> usize {
        if self.page_size == 0 {
            1
        } else {
            self.tensors.len().div_ceil(self.page_size)
        }
    }
    fn current_page_tensors(&self) -> &[TensorInfo] {
        let start = self.page * self.page_size;
        if start >= self.tensors.len() {
            return &[];
        }
        &self.tensors[start..(start + self.page_size).min(self.tensors.len())]
    }
    #[allow(dead_code)]
    fn navigate_next(&mut self) {
        self.current_tab = self.current_tab.next();
    }
}

#[derive(Debug, Clone)]
struct TensorStats {
    name: String,
    mean: f64,
    std_dev: f64,
    min: f64,
    max: f64,
    sparsity: f64,
}

// ---- Model Generation -------------------------------------------------------

const DTYPES: [(&str, usize); 4] = [("fp32", 4), ("fp16", 2), ("bf16", 2), ("int8", 1)];
const LAYER_PREFIXES: [&str; 6] = [
    "encoder.layer",
    "decoder.layer",
    "attention.qkv",
    "attention.output",
    "ffn.up",
    "ffn.down",
];

fn generate_tensors(rng: &mut impl Rng, count: usize) -> Vec<TensorInfo> {
    (0..count)
        .map(|i| {
            let name = format!(
                "{}.{}.weight",
                LAYER_PREFIXES[i % LAYER_PREFIXES.len()],
                i / LAYER_PREFIXES.len()
            );
            let dims = [768, 1024, 512, 256, 3072];
            let shape = vec![dims[i % 5], dims[(i + 2) % 5]];
            let (dtype, bpe) = DTYPES[rng.gen_range(0..DTYPES.len())];
            let n: usize = shape.iter().product();
            TensorInfo {
                name,
                shape,
                dtype: dtype.to_string(),
                size_bytes: n * bpe,
            }
        })
        .collect()
}

fn generate_tensor_stats(rng: &mut impl Rng, tensors: &[TensorInfo]) -> Vec<TensorStats> {
    tensors
        .iter()
        .map(|t| {
            let mean = rng.gen_range(-0.1..0.1);
            let std_dev = rng.gen_range(0.01..0.5);
            let rh = std_dev * rng.gen_range(2.5..4.0);
            TensorStats {
                name: t.name.clone(),
                mean,
                std_dev,
                min: mean - rh,
                max: mean + rh,
                sparsity: rng.gen_range(0.0..0.6),
            }
        })
        .collect()
}

// ---- Rendering Helpers ------------------------------------------------------

fn render_tab_bar(active: Tab) -> String {
    Tab::ALL
        .iter()
        .map(|tab| {
            if *tab == active {
                format!("[{}. *{}*]", tab.index(), tab.label())
            } else {
                format!(" {}. {} ", tab.index(), tab.label())
            }
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

fn render_hr(width: usize) -> String {
    std::iter::repeat(BOX_H).take(width).collect()
}

fn render_frame(title: &str, lines: &[String], width: usize) -> Vec<String> {
    let iw = width.saturating_sub(4);
    let tp = format!(" {} ", title);
    let mut out = vec![format!(
        "{}{}{}{}",
        BOX_TL,
        tp,
        render_hr(iw.saturating_sub(tp.len())),
        BOX_TR
    )];
    for line in lines {
        let padded = if line.len() < iw {
            format!("{}{}", line, " ".repeat(iw - line.len()))
        } else {
            line[..iw].to_string()
        };
        out.push(format!("{} {} {}", BOX_V, padded, BOX_V));
    }
    out.push(format!("{}{}{}", BOX_BL, render_hr(iw + 2), BOX_BR));
    out
}

fn render_bar(fraction: f64, width: usize) -> String {
    let filled = (fraction.clamp(0.0, 1.0) * width as f64) as usize;
    format!("[{}{}]", "#".repeat(filled), ".".repeat(width - filled))
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

fn format_params(params: usize) -> String {
    if params >= 1_000_000_000 {
        format!("{:.2}B", params as f64 / 1_000_000_000.0)
    } else if params >= 1_000_000 {
        format!("{:.2}M", params as f64 / 1_000_000.0)
    } else if params >= 1_000 {
        format!("{:.2}K", params as f64 / 1_000.0)
    } else {
        format!("{params}")
    }
}

// ---- Tab Renderers ----------------------------------------------------------

fn dtype_distribution(tensors: &[TensorInfo]) -> String {
    let mut counts = std::collections::HashMap::new();
    for t in tensors {
        *counts.entry(t.dtype.as_str()).or_insert(0usize) += 1;
    }
    let mut entries: Vec<_> = counts.into_iter().collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    entries
        .iter()
        .map(|(d, c)| format!("{d}: {c}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn render_overview(state: &AppState) -> Vec<String> {
    render_frame(
        "Overview",
        &[
            format!("Model Name:       {}", state.model_name),
            format!("Format:           APR v{}", state.format_version),
            format!("Model Version:    {}", state.model_version),
            format!("Created:          {}", state.creation_date),
            String::new(),
            format!("Total Parameters: {}", format_params(state.total_params())),
            format!(
                "Total Size:       {}",
                format_bytes(state.total_size_bytes())
            ),
            format!("Compression:      {:.2}x", state.compression_ratio()),
            format!("Tensor Count:     {}", state.tensors.len()),
            String::new(),
            "Dtype Distribution:".to_string(),
            format!("  {}", dtype_distribution(&state.tensors)),
        ],
        72,
    )
}

fn render_tensors(state: &AppState) -> Vec<String> {
    let pt = state.current_page_tensors();
    let si = state.page * state.page_size;
    let mut lines = vec![
        format!(
            "Page {}/{} ({} tensors total)",
            state.page + 1,
            state.total_pages(),
            state.tensors.len()
        ),
        String::new(),
        format!(
            "{:>3}  {:<36}  {:>12}  {:>5}  {:>10}",
            "#", "Name", "Shape", "Dtype", "Size"
        ),
        render_hr(72),
    ];
    for (i, t) in pt.iter().enumerate() {
        let n = if t.name.len() > 36 {
            format!("{}...", &t.name[..33])
        } else {
            t.name.clone()
        };
        lines.push(format!(
            "{:>3}  {:<36}  {:>12}  {:>5}  {:>10}",
            si + i + 1,
            n,
            format!("{:?}", t.shape),
            t.dtype,
            format_bytes(t.size_bytes)
        ));
    }
    if state.total_pages() > 1 {
        lines.push(String::new());
        lines.push("Navigation: [PgUp] Previous | [PgDn] Next".to_string());
    }
    render_frame("Tensors", &lines, 80)
}

fn compute_histogram_buckets<F>(
    stats: &[TensorStats],
    extractor: F,
    n: usize,
    lo: f64,
    hi: f64,
) -> Vec<(String, f64)>
where
    F: Fn(&TensorStats) -> f64,
{
    if n == 0 || stats.is_empty() {
        return Vec::new();
    }
    let bw = (hi - lo) / n as f64;
    let mut counts = vec![0usize; n];
    for s in stats {
        let idx = ((extractor(s) - lo) / bw).floor() as isize;
        counts[idx.clamp(0, (n - 1) as isize) as usize] += 1;
    }
    let max_c = counts.iter().copied().max().unwrap_or(1).max(1);
    counts
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let l = lo + i as f64 * bw;
            (
                format!("[{:>6.2}, {:>6.2})", l, l + bw),
                c as f64 / max_c as f64,
            )
        })
        .collect()
}

fn render_stats(stats: &[TensorStats]) -> Vec<String> {
    let mut lines = vec![
        format!(
            "{:<30}  {:>8}  {:>8}  {:>10}  {:>10}  {:>8}",
            "Tensor", "Mean", "StdDev", "Min", "Max", "Sparsity"
        ),
        render_hr(82),
    ];
    for s in stats.iter().take(10) {
        let n = if s.name.len() > 30 {
            format!("{}...", &s.name[..27])
        } else {
            s.name.clone()
        };
        lines.push(format!(
            "{:<30}  {:>8.4}  {:>8.4}  {:>10.4}  {:>10.4}  {:>7.1}%",
            n,
            s.mean,
            s.std_dev,
            s.min,
            s.max,
            s.sparsity * 100.0
        ));
    }
    if stats.len() > 10 {
        lines.push(format!("  ... and {} more tensors", stats.len() - 10));
    }
    lines.push(String::new());
    lines.push("Distribution Summary:".to_string());
    lines.push(String::new());
    lines.push("  Mean values:".to_string());
    for (label, frac) in &compute_histogram_buckets(stats, |s| s.mean, 5, -0.2, 0.2) {
        lines.push(format!("    {}: {}", label, render_bar(*frac, 30)));
    }
    lines.push(String::new());
    lines.push("  Sparsity:".to_string());
    for (label, frac) in &compute_histogram_buckets(stats, |s| s.sparsity, 5, 0.0, 1.0) {
        lines.push(format!("    {}: {}", label, render_bar(*frac, 30)));
    }
    render_frame("Stats", &lines, 88)
}

fn render_help() -> Vec<String> {
    render_frame(
        "Help",
        &[
            "Key Bindings:".into(),
            String::new(),
            "  Tab / Right Arrow    Next tab".into(),
            "  Shift+Tab / Left     Previous tab".into(),
            "  1-4                  Jump to tab by number".into(),
            "  PgDn / j             Next page (Tensors tab)".into(),
            "  PgUp / k             Previous page (Tensors tab)".into(),
            "  /                    Search tensors by name".into(),
            "  q / Esc              Quit".into(),
            String::new(),
            "Navigation:".into(),
            String::new(),
            "  The TUI has 4 tabs:".into(),
            "    1. Overview  - Model metadata and summary statistics".into(),
            "    2. Tensors   - Paginated list of all model tensors".into(),
            "    3. Stats     - Per-tensor statistics and histograms".into(),
            "    4. Help      - This help screen".into(),
        ],
        72,
    )
}

// ---- Navigation Simulation --------------------------------------------------

fn simulate_navigation(state: &mut AppState, stats: &[TensorStats]) {
    for (step, &tab) in [
        Tab::Overview,
        Tab::Tensors,
        Tab::Stats,
        Tab::Help,
        Tab::Overview,
    ]
    .iter()
    .enumerate()
    {
        state.current_tab = tab;
        println!("\n  Tab Bar: {}\n", render_tab_bar(state.current_tab));
        let frame = match state.current_tab {
            Tab::Overview => render_overview(state),
            Tab::Tensors => render_tensors(state),
            Tab::Stats => render_stats(stats),
            Tab::Help => render_help(),
        };
        for line in &frame {
            println!("  {line}");
        }
        if step < 4 {
            println!("\n  >>> Pressing [Tab] to navigate to next tab...");
        }
    }
    println!("\n  Navigation complete: cycled through all 4 tabs and back to Overview.");
}

// ---- Main -------------------------------------------------------------------

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("cli_apr_tui")?;
    println!("=== apr tui (Headless Mode) ===\n");
    let tensors = generate_tensors(ctx.rng(), NUM_TENSORS);
    let mut state = AppState {
        current_tab: Tab::Overview,
        model_name: "llama-3.2-1b.apr".to_string(),
        model_version: "1.0.0".to_string(),
        format_version: "2.0".to_string(),
        creation_date: "2026-02-25T12:00:00Z".to_string(),
        tensors,
        page: 0,
        page_size: PAGE_SIZE,
    };
    println!(
        "Model: {}  Tensors: {}  Params: {}  Size: {}",
        state.model_name,
        state.tensors.len(),
        format_params(state.total_params()),
        format_bytes(state.total_size_bytes())
    );
    let tensor_stats = generate_tensor_stats(ctx.rng(), &state.tensors);
    println!("\n--- Simulating TUI Navigation ---");
    simulate_navigation(&mut state, &tensor_stats);
    ctx.record_metric("tensor_count", state.tensors.len() as i64);
    ctx.record_metric("total_params", state.total_params() as i64);
    ctx.record_metric("total_size_bytes", state.total_size_bytes() as i64);
    ctx.record_float_metric("compression_ratio", state.compression_ratio());
    ctx.record_metric("total_pages", state.total_pages() as i64);
    println!();
    ctx.report()?;
    Ok(())
}

// ---- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn test_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    fn sample_state() -> AppState {
        AppState {
            current_tab: Tab::Overview,
            model_name: "test-model.apr".to_string(),
            model_version: "1.0.0".to_string(),
            format_version: "2.0".to_string(),
            creation_date: "2026-01-01T00:00:00Z".to_string(),
            tensors: vec![
                TensorInfo {
                    name: "layer.0.weight".to_string(),
                    shape: vec![768, 768],
                    dtype: "fp32".to_string(),
                    size_bytes: 768 * 768 * 4,
                },
                TensorInfo {
                    name: "layer.1.weight".to_string(),
                    shape: vec![768, 3072],
                    dtype: "fp16".to_string(),
                    size_bytes: 768 * 3072 * 2,
                },
            ],
            page: 0,
            page_size: PAGE_SIZE,
        }
    }

    #[test]
    fn test_tab_navigation_and_labels() {
        for (i, tab) in Tab::ALL.iter().enumerate() {
            assert_eq!(tab.index(), i + 1);
            assert!(!tab.label().is_empty());
        }
        let mut t = Tab::Overview;
        for expected in [Tab::Tensors, Tab::Stats, Tab::Help, Tab::Overview] {
            t = t.next();
            assert_eq!(t, expected);
        }
    }

    #[test]
    fn test_app_state_computations() {
        let state = sample_state();
        assert_eq!(state.total_params(), 768 * 768 + 768 * 3072);
        assert_eq!(state.total_size_bytes(), 768 * 768 * 4 + 768 * 3072 * 2);
        let expected_ratio = (state.total_params() * 4) as f64 / state.total_size_bytes() as f64;
        assert!((state.compression_ratio() - expected_ratio).abs() < 1e-10);
        assert_eq!(state.total_pages(), 1);
    }

    #[test]
    fn test_pagination() {
        let mut rng = test_rng();
        let mut state = AppState {
            tensors: generate_tensors(&mut rng, 24),
            page: 0,
            page_size: 10,
            current_tab: Tab::Tensors,
            model_name: "t".into(),
            model_version: "1.0".into(),
            format_version: "2.0".into(),
            creation_date: "2026-01-01".into(),
        };
        assert_eq!(state.total_pages(), 3);
        assert_eq!(state.current_page_tensors().len(), 10);
        state.page = 2;
        assert_eq!(state.current_page_tensors().len(), 4);
        state.page = 3;
        assert_eq!(state.current_page_tensors().len(), 0);
    }

    #[test]
    fn test_format_bytes_and_params() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1_048_576), "1.00 MB");
        assert_eq!(format_bytes(1_073_741_824), "1.00 GB");
        assert_eq!(format_params(500), "500");
        assert_eq!(format_params(1_500), "1.50K");
        assert_eq!(format_params(2_500_000), "2.50M");
        assert_eq!(format_params(7_000_000_000), "7.00B");
    }

    #[test]
    fn test_render_tab_bar_and_bar() {
        let bar = render_tab_bar(Tab::Tensors);
        assert!(bar.contains("*Tensors*"));
        assert!(!bar.contains("*Overview*"));
        assert_eq!(render_bar(0.0, 20).matches('#').count(), 0);
        assert_eq!(render_bar(1.0, 20).matches('#').count(), 20);
        assert_eq!(render_bar(2.0, 20).matches('#').count(), 20); // clamp
    }

    #[test]
    fn test_generate_tensors_and_stats() {
        let mut rng = test_rng();
        let tensors = generate_tensors(&mut rng, 12);
        assert_eq!(tensors.len(), 12);
        for t in &tensors {
            assert!(!t.name.is_empty());
            assert!(t.size_bytes > 0);
        }
        let mut rng2 = test_rng();
        let t2 = generate_tensors(&mut rng2, 5);
        let s1 = generate_tensor_stats(&mut rng, &tensors);
        let s2 = generate_tensor_stats(&mut rng2, &t2);
        assert_eq!(s1.len(), 12);
        assert_eq!(s2.len(), 5);
    }

    #[test]
    fn test_render_frame_structure() {
        let frame = render_frame("Test", &["Hello".to_string()], 40);
        assert!(frame[0].starts_with(BOX_TL));
        assert!(frame[0].contains("Test"));
        assert!(frame.last().unwrap().starts_with(BOX_BL));
    }

    #[test]
    fn test_render_overview_and_help() {
        let state = sample_state();
        let ov = render_overview(&state).join("\n");
        assert!(ov.contains("test-model.apr"));
        assert!(ov.contains("Compression"));
        let help = render_help().join("\n");
        assert!(help.contains("Tab / Right Arrow"));
        assert!(help.contains("Quit"));
    }

    #[test]
    fn test_compression_ratio_empty() {
        let state = AppState {
            current_tab: Tab::Overview,
            model_name: "empty".into(),
            model_version: "0.0".into(),
            format_version: "1.0".into(),
            creation_date: "2026-01-01".into(),
            tensors: vec![],
            page: 0,
            page_size: PAGE_SIZE,
        };
        assert!((state.compression_ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_histogram_buckets_empty() {
        assert!(compute_histogram_buckets(&[], |s| s.mean, 5, -1.0, 1.0).is_empty());
    }

    #[test]
    fn test_dtype_distribution_counts() {
        let tensors = vec![
            TensorInfo {
                name: "a".into(),
                shape: vec![10],
                dtype: "fp32".into(),
                size_bytes: 40,
            },
            TensorInfo {
                name: "b".into(),
                shape: vec![10],
                dtype: "fp32".into(),
                size_bytes: 40,
            },
            TensorInfo {
                name: "c".into(),
                shape: vec![10],
                dtype: "fp16".into(),
                size_bytes: 20,
            },
        ];
        let dist = dtype_distribution(&tensors);
        assert!(dist.contains("fp32: 2"));
        assert!(dist.contains("fp16: 1"));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use rand::SeedableRng;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_format_bytes_never_empty(bytes in 0usize..10_000_000_000) { prop_assert!(!format_bytes(bytes).is_empty()); }

        #[test]
        fn prop_render_bar_correct_length(fraction in -2.0f64..2.0, width in 1usize..50) { prop_assert_eq!(render_bar(fraction, width).len(), width + 2); }

        #[test]
        fn prop_total_pages_covers_all(n in 0usize..100, ps in 1usize..20) {
            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
            let state = AppState { current_tab: Tab::Overview, model_name: "t".into(), model_version: "1.0".into(),
                format_version: "1.0".into(), creation_date: "2026-01-01".into(), tensors: generate_tensors(&mut rng, n), page: 0, page_size: ps };
            prop_assert!(state.total_pages() * ps >= n);
        }
    }
}
