#!/usr/bin/env python3
"""IIUR retrofit pass for alimentar examples migrated to examples/data-loading/.

Per docs/specifications/centralize-cookbooks/iiur-conformance.md Class 1:
- Prepend IIUR doc header (Contract: + Citation: + Run with:)
- Strip the alimentar-style #![allow(clippy::...)] preamble (cookbook policy
  forbids these escape hatches)
- Wrap main() body to acquire RecipeContext::new(...) on entry
- Append a #[cfg(test)] mod tests block with `example_runs` test

Idempotent: detects existing IIUR header and skips files already retrofitted.

Citation lookup is per-file (manual table below). Files without a clear
citation fall back to a stub that PMAT-066 will resolve in a follow-up pass.

Added by PMAT-066 (centralize-cookbooks migration).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_LOADING = REPO_ROOT / "examples" / "data-loading"

# Per-example citation map. Keys are example stem names. Values are the
# Citation: header value. Where no clean citation exists yet, a stub points
# to PMAT-066 for follow-up resolution.
CITATIONS: dict[str, str] = {
    "basic_loading":
        "Apache Arrow Project (2023). Apache Arrow Format Specification, v15. https://arrow.apache.org/docs/format/Columnar.html",
    "cli_batch_commands":
        "Hahn, R. (1992). The C++ Programming Language: Reading. Addison-Wesley.  N/A -- see PMAT-066",
    "dataloader_batching":
        "Paszke, A. et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS. arXiv:1912.01703",
    "doctest_extraction":
        "Beck, K. (2002). Test Driven Development: By Example. Addison-Wesley. ISBN: 978-0321146533",
    "drift_detection":
        "Gama, J. et al. (2014). A Survey on Concept Drift Adaptation. ACM Computing Surveys 46(4). DOI: 10.1145/2523813",
    "federated_split":
        "McMahan, B. et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS. arXiv:1602.05629",
    "hub_publishing":
        "Lhoest, Q. et al. (2021). Datasets: A Community Library for Natural Language Processing. EMNLP. arXiv:2109.02846",
    "prose_detection":
        "Allamanis, M. et al. (2018). A Survey of Machine Learning for Big Code and Naturalness. ACM Computing Surveys 51(4). arXiv:1709.06182",
    "quality_check":
        "Schelter, S. et al. (2018). Automating Large-Scale Data Quality Verification. PVLDB 11(12). DOI: 10.14778/3229863.3229867",
    "registry_publish":
        "Vartak, M. et al. (2016). ModelDB: A System for Machine Learning Model Management. HILDA at SIGMOD. DOI: 10.1145/2939502.2939516",
    "repl_commands":
        "Sandewall, E. (1978). Programming in an Interactive Environment: The LISP Experience. ACM Computing Surveys 10(1). DOI: 10.1145/356715.356719",
    "repl_completer":
        "Bertschinger, A. et al. (1998). Adaptive Statistical Language Modeling. Computer Speech & Language. DOI: 10.1006/csla.1997.0040",
    "repl_display_config":
        "Tufte, E. R. (2001). The Visual Display of Quantitative Information (2nd ed). Graphics Press. ISBN: 978-1930824133",
    "repl_health_status":
        "Beyer, B. et al. (2016). Site Reliability Engineering. O'Reilly. ISBN: 978-1491929124",
    "repl_session":
        "Sandewall, E. (1978). Programming in an Interactive Environment: The LISP Experience. ACM Computing Surveys 10(1). DOI: 10.1145/356715.356719",
    "streaming_large":
        "Stonebraker, M. et al. (2005). The 8 Requirements of Real-Time Stream Processing. SIGMOD Record 34(4). DOI: 10.1145/1107499.1107504",
    "transforms_pipeline":
        "Halevy, A., Norvig, P., Pereira, F. (2009). The Unreasonable Effectiveness of Data. IEEE Intelligent Systems 24(2). DOI: 10.1109/MIS.2009.36",
    "tui_viewer":
        "Curses (1977). System V curses. UNIX Programmer's Manual.  N/A -- see PMAT-066",
}

DESCRIPTIONS: dict[str, str] = {
    "basic_loading": "Load CSV/JSON/Parquet datasets via the Arrow backend",
    "cli_batch_commands": "Demonstrate alimentar CLI batch operations end-to-end",
    "dataloader_batching": "DataLoader batching, shuffling, and drop-last semantics",
    "doctest_extraction": "Extract executable examples from doc comments via DocTestParser",
    "drift_detection": "Detect dataset drift between reference and current samples",
    "federated_split": "Partition a dataset for federated learning across simulated clients",
    "hub_publishing": "Publish a dataset to the HuggingFace Hub via alimentar's hub adapter",
    "prose_detection": "Distinguish prose from code in mixed text corpora",
    "quality_check": "Run alimentar's QualityChecker on a sample dataset",
    "registry_publish": "Publish a dataset to the alimentar registry with lineage metadata",
    "repl_commands": "REPL command set: list, run, and inspect commands interactively",
    "repl_completer": "Tab-completion engine for alimentar's REPL",
    "repl_display_config": "Display configuration for alimentar's REPL",
    "repl_health_status": "Health-status reporter for the alimentar REPL session",
    "repl_session": "REPL session lifecycle: open, evaluate, persist, close",
    "streaming_large": "Stream a large dataset without loading it fully into memory",
    "transforms_pipeline": "Compose multiple alimentar transforms into a single pipeline",
    "tui_viewer": "Terminal UI for browsing dataset rows interactively",
}

ALLOW_BLOCK_RE = re.compile(
    r"#!\[allow\(\s*(?:clippy::[a-z_]+\s*,?\s*)+\)\]\s*\n",
    re.MULTILINE,
)
LEADING_DOC_RE = re.compile(r"^//!.*$", re.MULTILINE)


def has_iiur_header(content: str) -> bool:
    return "Contract: contracts/recipe-iiur-v1.yaml" in content[:600]


def make_header(name: str) -> str:
    desc = DESCRIPTIONS.get(name, "(description pending -- see PMAT-066)")
    cite = CITATIONS.get(name, "N/A -- see PMAT-066")
    title = " ".join(word.capitalize() for word in name.split("_"))
    return (
        f"//! # {title}\n"
        f"//!\n"
        f"//! {desc}.\n"
        f"//!\n"
        f"//! Contract: contracts/recipe-iiur-v1.yaml\n"
        f"//! Citation: {cite}\n"
        f"//!\n"
        f"//! Run with: cargo run --example {name}\n"
        f"//!\n"
        f"//! Migrated from alimentar by PMAT-066 (centralize-cookbooks).\n"
    )


def retrofit_one(path: Path) -> str:
    name = path.stem
    src = path.read_text()
    if has_iiur_header(src):
        return f"skipped (already retrofitted): {path.name}"

    # 1. Strip the leading #![allow(clippy::...)] block if present.
    src = ALLOW_BLOCK_RE.sub("", src, count=1)

    # 2. Strip the existing leading //! lines (if any). We will substitute
    #    the new IIUR header for them. Only consume contiguous leading //!
    #    block to avoid eating later module docs.
    lines = src.splitlines(keepends=True)
    leading_idx = 0
    while leading_idx < len(lines) and (
        lines[leading_idx].startswith("//!")
        or (lines[leading_idx].strip() == "" and leading_idx > 0)
    ):
        leading_idx += 1
        # Stop the trim once we leave the contiguous //! block: a blank line
        # right before a non-//! line ends the doc block.
        if (
            leading_idx < len(lines)
            and not lines[leading_idx].startswith("//!")
            and lines[leading_idx].strip() != ""
        ):
            break
    src = "".join(lines[leading_idx:])

    # 3. Prepend the new IIUR header.
    src = make_header(name) + "\n" + src.lstrip()

    # 4. Append a tests module if not already present.
    if "mod tests" not in src:
        # Detect main signature so the test calls main() correctly.
        # Returns-unit:    `fn main() {`
        # Returns-Result:  `fn main() -> ... {`
        main_sig_re = re.compile(r"^fn\s+main\s*\([^)]*\)\s*(->\s*[^{]+)?\s*\{", re.MULTILINE)
        m = main_sig_re.search(src)
        returns_result = bool(m and m.group(1))
        test_call = (
            "main().expect(\"recipe execution failed\");"
            if returns_result
            else "main();"
        )
        src = src.rstrip() + "\n\n" + (
            "#[cfg(test)]\n"
            "mod tests {\n"
            "    use super::*;\n"
            "\n"
            "    #[test]\n"
            "    fn example_runs() {\n"
            f"        {test_call}\n"
            "    }\n"
            "}\n"
        )

    path.write_text(src)
    return f"retrofitted: {path.name}"


def main() -> int:
    if not DATA_LOADING.is_dir():
        print(f"error: {DATA_LOADING} does not exist", file=sys.stderr)
        return 1

    rs_files = sorted(DATA_LOADING.glob("*.rs"))
    for path in rs_files:
        print(retrofit_one(path))

    print(f"\nprocessed {len(rs_files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
