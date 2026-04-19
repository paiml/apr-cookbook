#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

pub const NUM_TENSORS: usize = 24;
pub const PAGE_SIZE: usize = 10;
pub const BOX_TL: char = '+';
pub const BOX_TR: char = '+';
pub const BOX_BL: char = '+';
pub const BOX_BR: char = '+';
pub const BOX_H: char = '-';
pub const BOX_V: char = '|';

// ---- Domain Types -----------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Overview,
    Tensors,
    Stats,
    Help,
}

impl Tab {
    pub const ALL: [Tab; 4] = [Tab::Overview, Tab::Tensors, Tab::Stats, Tab::Help];
    pub fn label(self) -> &'static str {
        match self {
            Self::Overview => "Overview",
            Self::Tensors => "Tensors",
            Self::Stats => "Stats",
            Self::Help => "Help",
        }
    }
    pub fn index(self) -> usize {
        match self {
            Self::Overview => 1,
            Self::Tensors => 2,
            Self::Stats => 3,
            Self::Help => 4,
        }
    }
    #[allow(dead_code)]
    pub fn next(self) -> Self {
        match self {
            Self::Overview => Self::Tensors,
            Self::Tensors => Self::Stats,
            Self::Stats => Self::Help,
            Self::Help => Self::Overview,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub size_bytes: usize,
}
impl TensorInfo {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Debug, Clone)]
pub struct AppState {
    pub current_tab: Tab,
    pub model_name: String,
    pub model_version: String,
    pub format_version: String,
    pub creation_date: String,
    pub tensors: Vec<TensorInfo>,
    pub page: usize,
    pub page_size: usize,
}

impl AppState {
    pub fn total_params(&self) -> usize {
        self.tensors.iter().map(TensorInfo::num_elements).sum()
    }
    pub fn total_size_bytes(&self) -> usize {
        self.tensors.iter().map(|t| t.size_bytes).sum()
    }
    pub fn compression_ratio(&self) -> f64 {
        let stored = self.total_size_bytes();
        if stored == 0 {
            return 1.0;
        }
        (self.total_params() * 4) as f64 / stored as f64
    }
    pub fn total_pages(&self) -> usize {
        if self.page_size == 0 {
            1
        } else {
            self.tensors.len().div_ceil(self.page_size)
        }
    }
    pub fn current_page_tensors(&self) -> &[TensorInfo] {
        let start = self.page * self.page_size;
        if start >= self.tensors.len() {
            return &[];
        }
        &self.tensors[start..(start + self.page_size).min(self.tensors.len())]
    }
    #[allow(dead_code)]
    pub fn navigate_next(&mut self) {
        self.current_tab = self.current_tab.next();
    }
}

#[derive(Debug, Clone)]
pub struct TensorStats {
    pub name: String,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub sparsity: f64,
}

// ---- Model Generation -------------------------------------------------------

pub const DTYPES: [(&str, usize); 4] = [("fp32", 4), ("fp16", 2), ("bf16", 2), ("int8", 1)];
pub const LAYER_PREFIXES: [&str; 6] = [
    "encoder.layer",
    "decoder.layer",
    "attention.qkv",
    "attention.output",
    "ffn.up",
    "ffn.down",
];

pub fn generate_tensors(rng: &mut impl Rng, count: usize) -> Vec<TensorInfo> {
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

pub fn generate_tensor_stats(rng: &mut impl Rng, tensors: &[TensorInfo]) -> Vec<TensorStats> {
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

pub fn render_tab_bar(active: Tab) -> String {
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

pub fn render_hr(width: usize) -> String {
    std::iter::repeat(BOX_H).take(width).collect()
}

pub fn render_frame(title: &str, lines: &[String], width: usize) -> Vec<String> {
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

pub fn render_bar(fraction: f64, width: usize) -> String {
    let filled = (fraction.clamp(0.0, 1.0) * width as f64) as usize;
    format!("[{}{}]", "#".repeat(filled), ".".repeat(width - filled))
}

pub fn format_bytes(bytes: usize) -> String {
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

pub fn format_params(params: usize) -> String {
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

pub fn dtype_distribution(tensors: &[TensorInfo]) -> String {
    let mut counts = std::collections::HashMap::new();
    for t in tensors {
        *counts.entry(t.dtype.as_str()).or_insert(0usize) += 1;
    }
    let mut entries: Vec<_> = counts.into_iter().collect();
    entries.sort_by_key(|b| std::cmp::Reverse(b.1));
    entries
        .iter()
        .map(|(d, c)| format!("{d}: {c}"))
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn render_overview(state: &AppState) -> Vec<String> {
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

pub fn render_tensors(state: &AppState) -> Vec<String> {
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

pub fn compute_histogram_buckets<F>(
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

pub fn render_stats(stats: &[TensorStats]) -> Vec<String> {
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

pub fn render_help() -> Vec<String> {
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

pub fn simulate_navigation(state: &mut AppState, stats: &[TensorStats]) {
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
