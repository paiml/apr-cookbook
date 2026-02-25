//! # Recipe: APR Model TUI (Headless Simulation)
//!
//! **Category**: CLI Tools
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (N/A)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Simulate a terminal UI for interactive model exploration, rendered in
//! headless mode. Mirrors `apr tui` with 4 tabs: Overview, Tensors, Stats,
//! and Help. Navigation between tabs is simulated without actual terminal
//! rendering.
//!
//! ## Run Command
//! ```bash
//! cargo run --example cli_apr_tui
//! ```

use apr_cookbook::prelude::*;
use rand::Rng;

// ============================================================================
// Constants
// ============================================================================

/// Number of tensors to generate for the simulated model.
const NUM_TENSORS: usize = 24;

/// Page size for the paginated tensor list.
const PAGE_SIZE: usize = 10;

/// Box-drawing characters for headless TUI rendering.
const BOX_TOP_LEFT: char = '+';
const BOX_TOP_RIGHT: char = '+';
const BOX_BOTTOM_LEFT: char = '+';
const BOX_BOTTOM_RIGHT: char = '+';
const BOX_HORIZONTAL: char = '-';
const BOX_VERTICAL: char = '|';

// ============================================================================
// Domain Types
// ============================================================================

/// Active tab in the TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Overview,
    Tensors,
    Stats,
    Help,
}

impl Tab {
    /// All tabs in display order.
    const ALL: [Tab; 4] = [Tab::Overview, Tab::Tensors, Tab::Stats, Tab::Help];

    /// Display label for the tab.
    fn label(self) -> &'static str {
        match self {
            Self::Overview => "Overview",
            Self::Tensors => "Tensors",
            Self::Stats => "Stats",
            Self::Help => "Help",
        }
    }

    /// Numeric index (1-based) for the tab.
    fn index(self) -> usize {
        match self {
            Self::Overview => 1,
            Self::Tensors => 2,
            Self::Stats => 3,
            Self::Help => 4,
        }
    }

    /// Return the next tab in circular order.
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

/// Information about a single tensor in the model.
#[derive(Debug, Clone)]
struct TensorInfo {
    name: String,
    shape: Vec<usize>,
    dtype: String,
    size_bytes: usize,
}

impl TensorInfo {
    /// Compute the number of elements from the shape.
    fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Complete application state for the TUI.
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
    /// Total number of parameters across all tensors.
    fn total_params(&self) -> usize {
        self.tensors.iter().map(TensorInfo::num_elements).sum()
    }

    /// Total size in bytes across all tensors.
    fn total_size_bytes(&self) -> usize {
        self.tensors.iter().map(|t| t.size_bytes).sum()
    }

    /// Compression ratio: raw fp32 size / actual stored size.
    fn compression_ratio(&self) -> f64 {
        let raw_fp32_size = self.total_params() * 4;
        let stored = self.total_size_bytes();
        if stored == 0 {
            return 1.0;
        }
        raw_fp32_size as f64 / stored as f64
    }

    /// Total number of pages for the tensor list.
    fn total_pages(&self) -> usize {
        if self.page_size == 0 {
            return 1;
        }
        self.tensors.len().div_ceil(self.page_size)
    }

    /// Get the tensor slice for the current page.
    fn current_page_tensors(&self) -> &[TensorInfo] {
        let start = self.page * self.page_size;
        let end = (start + self.page_size).min(self.tensors.len());
        if start >= self.tensors.len() {
            return &[];
        }
        &self.tensors[start..end]
    }

    /// Navigate to the next tab.
    #[allow(dead_code)]
    fn navigate_next(&mut self) {
        self.current_tab = self.current_tab.next();
    }
}

/// Per-tensor statistics for the Stats tab.
#[derive(Debug, Clone)]
struct TensorStats {
    name: String,
    mean: f64,
    std_dev: f64,
    min: f64,
    max: f64,
    sparsity: f64,
}

// ============================================================================
// Model Generation
// ============================================================================

/// Data type names with their byte sizes.
const DTYPES: [(&str, usize); 4] = [("fp32", 4), ("fp16", 2), ("bf16", 2), ("int8", 1)];

/// Layer name prefixes for realistic tensor naming.
const LAYER_PREFIXES: [&str; 6] = [
    "encoder.layer",
    "decoder.layer",
    "attention.qkv",
    "attention.output",
    "ffn.up",
    "ffn.down",
];

/// Generate a realistic set of tensors for a simulated model.
fn generate_tensors(rng: &mut impl Rng, count: usize) -> Vec<TensorInfo> {
    (0..count)
        .map(|i| {
            let prefix_idx = i % LAYER_PREFIXES.len();
            let layer_num = i / LAYER_PREFIXES.len();
            let name = format!("{}.{}.weight", LAYER_PREFIXES[prefix_idx], layer_num);

            // Generate a 2D shape with realistic dimensions
            let dim_a = [768, 1024, 512, 256, 3072][i % 5];
            let dim_b = [768, 1024, 512, 256, 3072][(i + 2) % 5];
            let shape = vec![dim_a, dim_b];

            let dtype_idx = rng.gen_range(0..DTYPES.len());
            let (dtype, bytes_per_elem) = DTYPES[dtype_idx];
            let n_elements: usize = shape.iter().product();
            let size_bytes = n_elements * bytes_per_elem;

            TensorInfo {
                name,
                shape,
                dtype: dtype.to_string(),
                size_bytes,
            }
        })
        .collect()
}

/// Generate deterministic per-tensor statistics.
fn generate_tensor_stats(rng: &mut impl Rng, tensors: &[TensorInfo]) -> Vec<TensorStats> {
    tensors
        .iter()
        .map(|t| {
            let mean = rng.gen_range(-0.1..0.1);
            let std_dev = rng.gen_range(0.01..0.5);
            let range_half = std_dev * rng.gen_range(2.5..4.0);
            let min_val = mean - range_half;
            let max_val = mean + range_half;
            let sparsity = rng.gen_range(0.0..0.6);

            TensorStats {
                name: t.name.clone(),
                mean,
                std_dev,
                min: min_val,
                max: max_val,
                sparsity,
            }
        })
        .collect()
}

// ============================================================================
// Rendering Helpers
// ============================================================================

/// Render the tab bar showing all tabs with the active tab highlighted.
fn render_tab_bar(active: Tab) -> String {
    let mut parts = Vec::new();
    for tab in &Tab::ALL {
        if *tab == active {
            parts.push(format!("[{}. *{}*]", tab.index(), tab.label()));
        } else {
            parts.push(format!(" {}. {} ", tab.index(), tab.label()));
        }
    }
    parts.join(" | ")
}

/// Render a horizontal rule of the given width.
fn render_hr(width: usize) -> String {
    std::iter::repeat(BOX_HORIZONTAL).take(width).collect()
}

/// Render a framed box around content lines.
fn render_frame(title: &str, lines: &[String], width: usize) -> Vec<String> {
    let mut output = Vec::new();
    let inner_width = width.saturating_sub(4);

    // Top border
    let title_padded = format!(" {} ", title);
    let remaining = inner_width.saturating_sub(title_padded.len());
    let top = format!(
        "{}{}{}{}",
        BOX_TOP_LEFT,
        title_padded,
        render_hr(remaining),
        BOX_TOP_RIGHT
    );
    output.push(top);

    // Content
    for line in lines {
        let padded = if line.len() < inner_width {
            let padding = inner_width - line.len();
            format!("{}{}", line, " ".repeat(padding))
        } else {
            line[..inner_width].to_string()
        };
        output.push(format!("{} {} {}", BOX_VERTICAL, padded, BOX_VERTICAL));
    }

    // Bottom border
    let bottom = format!(
        "{}{}{}",
        BOX_BOTTOM_LEFT,
        render_hr(inner_width + 2),
        BOX_BOTTOM_RIGHT
    );
    output.push(bottom);

    output
}

/// Render an ASCII histogram bar of the given fraction (0.0 to 1.0).
fn render_bar(fraction: f64, width: usize) -> String {
    let clamped = fraction.clamp(0.0, 1.0);
    let filled = (clamped * width as f64) as usize;
    let empty = width - filled;
    let mut bar = String::with_capacity(width + 2);
    bar.push('[');
    for _ in 0..filled {
        bar.push('#');
    }
    for _ in 0..empty {
        bar.push('.');
    }
    bar.push(']');
    bar
}

/// Format a byte count as a human-readable string.
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

/// Format a parameter count with suffix (K, M, B).
fn format_params(params: usize) -> String {
    if params >= 1_000_000_000 {
        format!("{:.2}B", params as f64 / 1_000_000_000.0)
    } else if params >= 1_000_000 {
        format!("{:.2}M", params as f64 / 1_000_000.0)
    } else if params >= 1_000 {
        format!("{:.2}K", params as f64 / 1_000.0)
    } else {
        format!("{}", params)
    }
}

// ============================================================================
// Tab Renderers
// ============================================================================

/// Render the Overview tab content.
fn render_overview(state: &AppState) -> Vec<String> {
    let lines = vec![
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
    ];
    render_frame("Overview", &lines, 72)
}

/// Compute dtype distribution string for the overview tab.
fn dtype_distribution(tensors: &[TensorInfo]) -> String {
    let mut counts = std::collections::HashMap::new();
    for t in tensors {
        *counts.entry(t.dtype.as_str()).or_insert(0usize) += 1;
    }
    let mut entries: Vec<_> = counts.into_iter().collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    entries
        .iter()
        .map(|(dtype, count)| format!("{}: {}", dtype, count))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Render the Tensors tab content (paginated).
fn render_tensors(state: &AppState) -> Vec<String> {
    let page_tensors = state.current_page_tensors();
    let start_idx = state.page * state.page_size;

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
        format!("{}", render_hr(72)),
    ];

    for (i, tensor) in page_tensors.iter().enumerate() {
        let idx = start_idx + i + 1;
        let shape_str = format!("{:?}", tensor.shape);
        let truncated_name = if tensor.name.len() > 36 {
            format!("{}...", &tensor.name[..33])
        } else {
            tensor.name.clone()
        };
        lines.push(format!(
            "{:>3}  {:<36}  {:>12}  {:>5}  {:>10}",
            idx,
            truncated_name,
            shape_str,
            tensor.dtype,
            format_bytes(tensor.size_bytes)
        ));
    }

    if state.total_pages() > 1 {
        lines.push(String::new());
        lines.push("Navigation: [PgUp] Previous | [PgDn] Next".to_string());
    }

    render_frame("Tensors", &lines, 80)
}

/// Render the Stats tab content.
fn render_stats(stats: &[TensorStats]) -> Vec<String> {
    let mut lines = vec![
        format!(
            "{:<30}  {:>8}  {:>8}  {:>10}  {:>10}  {:>8}",
            "Tensor", "Mean", "StdDev", "Min", "Max", "Sparsity"
        ),
        format!("{}", render_hr(82)),
    ];

    // Show up to 10 tensors in the stats view
    let display_count = stats.len().min(10);
    for stat in stats.iter().take(display_count) {
        let truncated_name = if stat.name.len() > 30 {
            format!("{}...", &stat.name[..27])
        } else {
            stat.name.clone()
        };
        lines.push(format!(
            "{:<30}  {:>8.4}  {:>8.4}  {:>10.4}  {:>10.4}  {:>7.1}%",
            truncated_name,
            stat.mean,
            stat.std_dev,
            stat.min,
            stat.max,
            stat.sparsity * 100.0
        ));
    }

    if stats.len() > display_count {
        lines.push(format!(
            "  ... and {} more tensors",
            stats.len() - display_count
        ));
    }

    // Summary histograms
    lines.push(String::new());
    lines.push("Distribution Summary (ASCII Histogram):".to_string());
    lines.push(String::new());

    // Mean distribution histogram
    lines.push("  Mean values:".to_string());
    let mean_buckets = compute_histogram_buckets(stats, |s| s.mean, 5, -0.2, 0.2);
    for (label, fraction) in &mean_buckets {
        lines.push(format!("    {}: {}", label, render_bar(*fraction, 30)));
    }

    // Sparsity distribution histogram
    lines.push(String::new());
    lines.push("  Sparsity:".to_string());
    let sparsity_buckets = compute_histogram_buckets(stats, |s| s.sparsity, 5, 0.0, 1.0);
    for (label, fraction) in &sparsity_buckets {
        lines.push(format!("    {}: {}", label, render_bar(*fraction, 30)));
    }

    render_frame("Stats", &lines, 88)
}

/// Compute histogram buckets for a given extractor function.
fn compute_histogram_buckets<F>(
    stats: &[TensorStats],
    extractor: F,
    num_buckets: usize,
    range_min: f64,
    range_max: f64,
) -> Vec<(String, f64)>
where
    F: Fn(&TensorStats) -> f64,
{
    if num_buckets == 0 || stats.is_empty() {
        return Vec::new();
    }

    let bucket_width = (range_max - range_min) / num_buckets as f64;
    let mut counts = vec![0usize; num_buckets];

    for stat in stats {
        let val = extractor(stat);
        let bucket_idx = ((val - range_min) / bucket_width).floor() as isize;
        let clamped = bucket_idx.clamp(0, (num_buckets - 1) as isize) as usize;
        counts[clamped] += 1;
    }

    let max_count = counts.iter().copied().max().unwrap_or(1).max(1);

    counts
        .iter()
        .enumerate()
        .map(|(i, &count)| {
            let lo = range_min + i as f64 * bucket_width;
            let hi = lo + bucket_width;
            let label = format!("[{:>6.2}, {:>6.2})", lo, hi);
            let fraction = count as f64 / max_count as f64;
            (label, fraction)
        })
        .collect()
}

/// Render the Help tab content.
fn render_help() -> Vec<String> {
    let lines = vec![
        "Key Bindings:".to_string(),
        String::new(),
        "  Tab / Right Arrow    Next tab".to_string(),
        "  Shift+Tab / Left     Previous tab".to_string(),
        "  1-4                  Jump to tab by number".to_string(),
        "  PgDn / j             Next page (Tensors tab)".to_string(),
        "  PgUp / k             Previous page (Tensors tab)".to_string(),
        "  /                    Search tensors by name".to_string(),
        "  q / Esc              Quit".to_string(),
        String::new(),
        "Navigation:".to_string(),
        String::new(),
        "  The TUI has 4 tabs:".to_string(),
        "    1. Overview  - Model metadata and summary statistics".to_string(),
        "    2. Tensors   - Paginated list of all model tensors".to_string(),
        "    3. Stats     - Per-tensor statistics and histograms".to_string(),
        "    4. Help      - This help screen".to_string(),
        String::new(),
        "  Tensor pages show 10 tensors at a time. Use PgUp/PgDn".to_string(),
        "  to navigate between pages when viewing large models.".to_string(),
        String::new(),
        "  apr tui [MODEL_PATH]  -- Launch with a specific model".to_string(),
        "  apr tui --headless    -- Print all tabs without TUI".to_string(),
    ];
    render_frame("Help", &lines, 72)
}

// ============================================================================
// Navigation Simulation
// ============================================================================

/// Simulate navigating through all tabs and print output.
fn simulate_navigation(state: &mut AppState, stats: &[TensorStats]) {
    // We visit all 4 tabs in order, then cycle back to tab 1
    let navigation_sequence = [
        Tab::Overview,
        Tab::Tensors,
        Tab::Stats,
        Tab::Help,
        Tab::Overview,
    ];

    for (step, &target_tab) in navigation_sequence.iter().enumerate() {
        state.current_tab = target_tab;

        println!();
        println!("  Tab Bar: {}", render_tab_bar(state.current_tab));
        println!();

        let frame_lines = match state.current_tab {
            Tab::Overview => render_overview(state),
            Tab::Tensors => render_tensors(state),
            Tab::Stats => render_stats(stats),
            Tab::Help => render_help(),
        };

        for line in &frame_lines {
            println!("  {}", line);
        }

        if step < navigation_sequence.len() - 1 {
            println!();
            println!("  >>> Pressing [Tab] to navigate to next tab...");
        }
    }

    println!();
    println!("  Navigation complete: cycled through all 4 tabs and back to Overview.");
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("cli_apr_tui")?;

    println!("=== apr tui (Headless Mode) ===");
    println!();

    // =========================================================================
    // Section 1: Generate simulated model data
    // =========================================================================

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

    println!("Model: {}", state.model_name);
    println!("Tensors: {}", state.tensors.len());
    println!("Total Parameters: {}", format_params(state.total_params()));
    println!("Total Size: {}", format_bytes(state.total_size_bytes()));

    // =========================================================================
    // Section 2: Generate tensor statistics
    // =========================================================================

    let tensor_stats = generate_tensor_stats(ctx.rng(), &state.tensors);

    // =========================================================================
    // Section 3: Simulate tab navigation
    // =========================================================================

    println!();
    println!("--- Simulating TUI Navigation ---");

    simulate_navigation(&mut state, &tensor_stats);

    // =========================================================================
    // Section 4: Record metrics
    // =========================================================================

    ctx.record_metric("tensor_count", state.tensors.len() as i64);
    ctx.record_metric("total_params", state.total_params() as i64);
    ctx.record_metric("total_size_bytes", state.total_size_bytes() as i64);
    ctx.record_float_metric("compression_ratio", state.compression_ratio());
    ctx.record_metric("total_pages", state.total_pages() as i64);

    println!();
    ctx.report()?;
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn test_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    fn sample_tensors() -> Vec<TensorInfo> {
        vec![
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
        ]
    }

    fn sample_state() -> AppState {
        AppState {
            current_tab: Tab::Overview,
            model_name: "test-model.apr".to_string(),
            model_version: "1.0.0".to_string(),
            format_version: "2.0".to_string(),
            creation_date: "2026-01-01T00:00:00Z".to_string(),
            tensors: sample_tensors(),
            page: 0,
            page_size: PAGE_SIZE,
        }
    }

    #[test]
    fn test_tab_labels() {
        assert_eq!(Tab::Overview.label(), "Overview");
        assert_eq!(Tab::Tensors.label(), "Tensors");
        assert_eq!(Tab::Stats.label(), "Stats");
        assert_eq!(Tab::Help.label(), "Help");
    }

    #[test]
    fn test_tab_circular_navigation() {
        let mut tab = Tab::Overview;
        tab = tab.next(); // -> Tensors
        assert_eq!(tab, Tab::Tensors);
        tab = tab.next(); // -> Stats
        assert_eq!(tab, Tab::Stats);
        tab = tab.next(); // -> Help
        assert_eq!(tab, Tab::Help);
        tab = tab.next(); // -> Overview (wrap)
        assert_eq!(tab, Tab::Overview);
    }

    #[test]
    fn test_tensor_info_num_elements() {
        let tensor = TensorInfo {
            name: "w".to_string(),
            shape: vec![768, 1024],
            dtype: "fp32".to_string(),
            size_bytes: 768 * 1024 * 4,
        };
        assert_eq!(tensor.num_elements(), 768 * 1024);
    }

    #[test]
    fn test_app_state_total_params() {
        let state = sample_state();
        let expected = 768 * 768 + 768 * 3072;
        assert_eq!(state.total_params(), expected);
    }

    #[test]
    fn test_app_state_total_size() {
        let state = sample_state();
        let expected = 768 * 768 * 4 + 768 * 3072 * 2;
        assert_eq!(state.total_size_bytes(), expected);
    }

    #[test]
    fn test_app_state_compression_ratio() {
        let state = sample_state();
        let raw_fp32 = state.total_params() * 4;
        let stored = state.total_size_bytes();
        let expected = raw_fp32 as f64 / stored as f64;
        assert!((state.compression_ratio() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_pagination_total_pages() {
        let mut state = sample_state();
        // 2 tensors, page_size=10 -> 1 page
        assert_eq!(state.total_pages(), 1);

        // Generate more tensors
        let mut rng = test_rng();
        state.tensors = generate_tensors(&mut rng, 24);
        state.page_size = 10;
        // 24 tensors / 10 per page = 3 pages
        assert_eq!(state.total_pages(), 3);
    }

    #[test]
    fn test_pagination_current_page_tensors() {
        let mut rng = test_rng();
        let mut state = AppState {
            current_tab: Tab::Tensors,
            model_name: "test".to_string(),
            model_version: "1.0".to_string(),
            format_version: "2.0".to_string(),
            creation_date: "2026-01-01".to_string(),
            tensors: generate_tensors(&mut rng, 24),
            page: 0,
            page_size: 10,
        };
        assert_eq!(state.current_page_tensors().len(), 10);
        state.page = 1;
        assert_eq!(state.current_page_tensors().len(), 10);
        state.page = 2;
        assert_eq!(state.current_page_tensors().len(), 4);
        state.page = 3;
        assert_eq!(state.current_page_tensors().len(), 0);
    }

    #[test]
    fn test_format_bytes_ranges() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1_048_576), "1.00 MB");
        assert_eq!(format_bytes(1_073_741_824), "1.00 GB");
    }

    #[test]
    fn test_format_params_ranges() {
        assert_eq!(format_params(500), "500");
        assert_eq!(format_params(1_500), "1.50K");
        assert_eq!(format_params(2_500_000), "2.50M");
        assert_eq!(format_params(7_000_000_000), "7.00B");
    }

    #[test]
    fn test_render_tab_bar_highlights_active() {
        let bar = render_tab_bar(Tab::Tensors);
        assert!(
            bar.contains("*Tensors*"),
            "active tab should be highlighted"
        );
        assert!(
            !bar.contains("*Overview*"),
            "inactive tab should not be highlighted"
        );
    }

    #[test]
    fn test_render_bar_boundaries() {
        let empty = render_bar(0.0, 20);
        assert_eq!(empty.len(), 22); // 20 + brackets
        assert!(empty.starts_with('['));
        assert!(empty.ends_with(']'));
        assert_eq!(empty.matches('#').count(), 0);

        let full = render_bar(1.0, 20);
        assert_eq!(full.matches('#').count(), 20);

        // Clamp above 1.0
        let over = render_bar(2.0, 20);
        assert_eq!(over.matches('#').count(), 20);
    }

    #[test]
    fn test_generate_tensors_count_and_names() {
        let mut rng = test_rng();
        let tensors = generate_tensors(&mut rng, 12);
        assert_eq!(tensors.len(), 12);
        for t in &tensors {
            assert!(!t.name.is_empty(), "tensor name should not be empty");
            assert!(!t.shape.is_empty(), "tensor shape should not be empty");
            assert!(t.size_bytes > 0, "tensor size should be positive");
        }
    }

    #[test]
    fn test_generate_tensor_stats_deterministic() {
        let mut rng1 = test_rng();
        let mut rng2 = test_rng();
        let tensors1 = generate_tensors(&mut rng1, 5);
        let stats1 = generate_tensor_stats(&mut rng1, &tensors1);
        let tensors2 = generate_tensors(&mut rng2, 5);
        let stats2 = generate_tensor_stats(&mut rng2, &tensors2);

        assert_eq!(stats1.len(), stats2.len());
        for (a, b) in stats1.iter().zip(stats2.iter()) {
            assert!((a.mean - b.mean).abs() < 1e-10, "means should match");
            assert!(
                (a.std_dev - b.std_dev).abs() < 1e-10,
                "std_devs should match"
            );
        }
    }

    #[test]
    fn test_compute_histogram_buckets_empty() {
        let buckets = compute_histogram_buckets(&[], |s| s.mean, 5, -1.0, 1.0);
        assert!(buckets.is_empty());
    }

    #[test]
    fn test_navigate_next_cycles_all_tabs() {
        let mut state = sample_state();
        assert_eq!(state.current_tab, Tab::Overview);
        state.navigate_next();
        assert_eq!(state.current_tab, Tab::Tensors);
        state.navigate_next();
        assert_eq!(state.current_tab, Tab::Stats);
        state.navigate_next();
        assert_eq!(state.current_tab, Tab::Help);
        state.navigate_next();
        assert_eq!(state.current_tab, Tab::Overview);
    }

    #[test]
    fn test_render_overview_contains_model_name() {
        let state = sample_state();
        let lines = render_overview(&state);
        let combined: String = lines.join("\n");
        assert!(
            combined.contains("test-model.apr"),
            "overview should contain model name"
        );
        assert!(
            combined.contains("Compression"),
            "overview should show compression"
        );
    }

    #[test]
    fn test_render_help_contains_keybindings() {
        let lines = render_help();
        let combined: String = lines.join("\n");
        assert!(
            combined.contains("Tab / Right Arrow"),
            "help should contain navigation keys"
        );
        assert!(combined.contains("Quit"), "help should mention quit");
    }

    #[test]
    fn test_dtype_distribution_counts() {
        let tensors = vec![
            TensorInfo {
                name: "a".to_string(),
                shape: vec![10],
                dtype: "fp32".to_string(),
                size_bytes: 40,
            },
            TensorInfo {
                name: "b".to_string(),
                shape: vec![10],
                dtype: "fp32".to_string(),
                size_bytes: 40,
            },
            TensorInfo {
                name: "c".to_string(),
                shape: vec![10],
                dtype: "fp16".to_string(),
                size_bytes: 20,
            },
        ];
        let dist = dtype_distribution(&tensors);
        assert!(dist.contains("fp32: 2"), "should show fp32 count");
        assert!(dist.contains("fp16: 1"), "should show fp16 count");
    }

    #[test]
    fn test_compression_ratio_zero_size() {
        let state = AppState {
            current_tab: Tab::Overview,
            model_name: "empty".to_string(),
            model_version: "0.0".to_string(),
            format_version: "1.0".to_string(),
            creation_date: "2026-01-01".to_string(),
            tensors: vec![],
            page: 0,
            page_size: PAGE_SIZE,
        };
        // Empty tensors: total_size = 0, should return 1.0 (no division by zero)
        assert!((state.compression_ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_render_frame_structure() {
        let lines = vec!["Hello".to_string(), "World".to_string()];
        let frame = render_frame("Test", &lines, 40);
        assert!(frame.len() >= 4, "frame should have top, content, bottom");
        assert!(frame[0].starts_with(BOX_TOP_LEFT));
        assert!(frame[0].contains("Test"));
        let last = frame.last().expect("frame should not be empty");
        assert!(last.starts_with(BOX_BOTTOM_LEFT));
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
        fn prop_format_bytes_never_empty(bytes in 0usize..10_000_000_000) {
            let formatted = format_bytes(bytes);
            prop_assert!(!formatted.is_empty());
        }

        #[test]
        fn prop_format_params_never_empty(params in 0usize..10_000_000_000) {
            let formatted = format_params(params);
            prop_assert!(!formatted.is_empty());
        }

        #[test]
        fn prop_render_bar_correct_length(fraction in -2.0f64..2.0, width in 1usize..50) {
            let bar = render_bar(fraction, width);
            // +2 for brackets
            prop_assert_eq!(bar.len(), width + 2);
        }

        #[test]
        fn prop_tab_next_always_valid(start in 0usize..4) {
            let tab = Tab::ALL[start];
            let next = tab.next();
            // next should always be a valid tab
            prop_assert!(Tab::ALL.contains(&next));
        }

        #[test]
        fn prop_total_pages_covers_all_tensors(
            n_tensors in 0usize..100,
            page_size in 1usize..20,
        ) {
            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
            let tensors = generate_tensors(&mut rng, n_tensors);
            let state = AppState {
                current_tab: Tab::Overview,
                model_name: "test".to_string(),
                model_version: "1.0".to_string(),
                format_version: "1.0".to_string(),
                creation_date: "2026-01-01".to_string(),
                tensors,
                page: 0,
                page_size,
            };
            let pages = state.total_pages();
            // total_pages * page_size >= n_tensors
            prop_assert!(pages * page_size >= n_tensors);
            // And pages is minimal
            if n_tensors > 0 {
                prop_assert!(pages > 0);
            }
        }
    }
}
