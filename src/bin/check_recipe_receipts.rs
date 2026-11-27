//! Required check: every recipe carries a PASS receipt from the CURRENT apr release (#442).
//!
//! Rust rewrite (#472) of the former `scripts/check_recipe_receipts.py`. Same rules, same
//! output lines, same exit codes. A receipt nobody reads is an artifact, not evidence. This
//! check reads the committed receipts that `scripts/run_recipes.py` writes (#439) and fails
//! unless, for EVERY recipe under recipes/apr/ and EVERY host that recipe declares:
//!
//!   receipts/<CURRENT>/<host>/<id>.json exists,
//!   its verdict is PASS (FAIL and NOT_RUN are not passes),
//!   its apr_version names the CURRENT release's binary, and
//!   its recipe_sha256 equals the recipe file's sha256 now.
//!
//! <CURRENT> is the one line in receipts/CURRENT, e.g. `0.69.1-d8a6df53a`. It never runs a
//! model: CI holds none. It judges committed evidence only.
//!
//! Usage: cargo run -q --bin check_recipe_receipts -- [--recipes DIR] [--receipts DIR]
//!        cargo run -q --bin check_recipe_receipts -- --self-test
//! Exit: 0 every recipe covered | 1 a gap or a bad receipt | 2 usage / no CURRENT / no recipes

use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

fn sha256_file(p: &Path) -> std::io::Result<String> {
    let bytes = fs::read(p)?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

fn collect_yaml(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let p = entry?.path();
        if p.is_dir() {
            collect_yaml(&p, out)?;
        } else if p.extension().is_some_and(|e| e == "yaml") {
            out.push(p);
        }
    }
    Ok(())
}

/// Python-style repr of a JSON value, for the messages that quote a receipt field.
fn py_repr(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => "None".to_string(),
        serde_json::Value::String(s) => format!("'{s}'"),
        other => other.to_string(),
    }
}

/// Python `%s` of a JSON value: strings bare, null as "None".
fn py_str(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => "None".to_string(),
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// The CURRENT token: (`<version>-<sha>` as written, its `<sha>` part). Err is the REFUSED line.
fn read_current(receipts_dir: &Path) -> Result<(String, String), String> {
    let cur_file = receipts_dir.join("CURRENT");
    let current = fs::read_to_string(&cur_file)
        .map(|s| s.trim().to_string())
        .map_err(|_| {
            format!(
                "REFUSED: no {} -- the check cannot know which release's receipts are required",
                cur_file.display()
            )
        })?;
    if current.is_empty() || !current.contains('-') {
        return Err(format!(
            "REFUSED: {} holds '{}', not a <version>-<sha> token",
            cur_file.display(),
            current
        ));
    }
    let sha = current.split_once('-').map_or("", |(_, s)| s).to_string();
    Ok((current, sha))
}

fn recipe_files(recipes_dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if collect_yaml(recipes_dir, &mut files).is_err() {
        files.clear();
    }
    files.sort();
    files
}

/// One recipe: (id, declared hosts, sha256 of its file now). Err is the problem line.
fn load_recipe(f: &Path) -> Result<(String, Vec<String>, String), String> {
    let text =
        fs::read_to_string(f).map_err(|e| format!("{}: recipe unreadable ({e})", f.display()))?;
    let r: serde_yaml::Value = serde_yaml::from_str(&text)
        .map_err(|e| format!("{}: recipe is not YAML ({e})", f.display()))?;
    let rid = r["id"].as_str().unwrap_or("").to_string();
    if rid.is_empty() {
        return Err(format!("{}: recipe has no id", f.display()));
    }
    let entries = r["hosts"].as_sequence().cloned().unwrap_or_default();
    // A non-string entry (a stray `-`, a bare number) must refuse the recipe, never shrink
    // the cell count: Python died on it (rc 1), so this is the same fail-closed outcome.
    if let Some(i) = entries.iter().position(|h| !h.is_string()) {
        return Err(format!(
            "{rid}: host {i} is not a string -- refusing the recipe"
        ));
    }
    let hosts: Vec<String> = entries
        .iter()
        .filter_map(|h| h.as_str().map(String::from))
        .collect();
    if hosts.is_empty() {
        return Err(format!(
            "{rid}: declares no hosts -- nothing could ever vouch for it"
        ));
    }
    let want = sha256_file(f).map_err(|e| format!("{}: recipe unreadable ({e})", f.display()))?;
    Ok((rid, hosts, want))
}

/// The receipt-level rules for one recipe x host cell. Empty means the cell is covered.
fn check_receipt(
    rec: &serde_json::Value,
    rid: &str,
    h: &str,
    sha: &str,
    want_sha: &str,
) -> Vec<String> {
    let mut bad = Vec::new();
    if rec["verdict"].as_str() != Some("PASS") {
        let reasons: Vec<String> = rec["reasons"]
            .as_array()
            .map(|a| a.iter().map(py_str).collect())
            .unwrap_or_default();
        bad.push(format!(
            "{rid} @ {h}: verdict {} -- {}",
            py_str(&rec["verdict"]),
            reasons.join("; ")
        ));
    }
    let apr = rec["apr_version"].as_str().unwrap_or("");
    if !apr.contains(sha) {
        bad.push(format!(
            "{rid} @ {h}: receipt names {}, not the CURRENT binary {sha}",
            py_repr(&rec["apr_version"])
        ));
    }
    if rec["recipe_sha256"].as_str() != Some(want_sha) {
        let got: String = rec["recipe_sha256"]
            .as_str()
            .filter(|s| !s.is_empty())
            .unwrap_or("none")
            .chars()
            .take(12)
            .collect();
        let want12: String = want_sha.chars().take(12).collect();
        bad.push(format!(
            "{rid} @ {h}: the recipe changed since this receipt (receipt {got}, file {want12}) -- re-run it"
        ));
    }
    bad
}

/// One recipe x host cell: its receipt must exist, parse, and pass every receipt rule.
fn check_cell(
    rid: &str,
    h: &str,
    receipts_dir: &Path,
    current: &str,
    sha: &str,
    want_sha: &str,
) -> Vec<String> {
    let rp = receipts_dir
        .join(current)
        .join(h)
        .join(format!("{rid}.json"));
    if !rp.is_file() {
        return vec![format!(
            "{rid} @ {h}: no receipt from {current} ({})",
            rp.display()
        )];
    }
    let parsed: Result<serde_json::Value, String> = fs::read_to_string(&rp)
        .map_err(|e| e.to_string())
        .and_then(|t| serde_json::from_str(&t).map_err(|e| e.to_string()));
    match parsed {
        Ok(rec) => check_receipt(&rec, rid, h, sha, want_sha),
        Err(e) => vec![format!("{rid} @ {h}: receipt is not JSON ({e})")],
    }
}

/// One recipe file: (cells checked, problems).
fn check_recipe(f: &Path, receipts_dir: &Path, current: &str, sha: &str) -> (usize, Vec<String>) {
    let (rid, hosts, want_sha) = match load_recipe(f) {
        Ok(v) => v,
        Err(problem) => return (0, vec![problem]),
    };
    let bad = hosts
        .iter()
        .flat_map(|h| check_cell(&rid, h, receipts_dir, current, sha, &want_sha))
        .collect();
    (hosts.len(), bad)
}

/// Returns (exit code, output lines). The caller prints them.
fn check(recipes_dir: &Path, receipts_dir: &Path) -> (i32, Vec<String>) {
    let (current, sha) = match read_current(receipts_dir) {
        Ok(v) => v,
        Err(line) => return (2, vec![line]),
    };
    let files = recipe_files(recipes_dir);
    if files.is_empty() {
        let line = format!(
            "REFUSED: no recipes under {} -- zero recipes is broken wiring, not a pass",
            recipes_dir.display()
        );
        return (2, vec![line]);
    }
    let mut bad: Vec<String> = Vec::new();
    let mut checked = 0usize;
    for f in &files {
        let (cells, problems) = check_recipe(f, receipts_dir, &current, &sha);
        checked += cells;
        bad.extend(problems);
    }
    let mut lines: Vec<String> = bad.iter().map(|b| format!("FAIL  {b}")).collect();
    lines.push(format!(
        "{}  {checked} recipe x host cell(s) checked against {current}, {} problem(s)",
        if bad.is_empty() { "PASS" } else { "FAIL" },
        bad.len()
    ));
    (i32::from(!bad.is_empty()), lines)
}

fn write_receipt(
    xdir: &Path,
    cur: &str,
    host: &str,
    rf: &Path,
    over: &[(&str, serde_json::Value)],
) {
    let d = xdir.join(cur).join(host);
    fs::create_dir_all(&d).expect("receipt dir");
    let mut rec = serde_json::json!({
        "recipe": "r1",
        "verdict": "PASS",
        "reasons": [],
        "apr_version": "apr 0.69.1 (abc1234)",
        "recipe_sha256": sha256_file(rf).expect("recipe sha"),
    });
    for (k, v) in over {
        rec[*k] = v.clone();
    }
    fs::write(d.join("r1.json"), rec.to_string()).expect("write receipt");
}

/// The self-test's own fixture: a `Fixture` owns the temp tree and rewrites it per case.
struct Fixture {
    _tmp: tempfile::TempDir,
    rdir: PathBuf,
    xdir: PathBuf,
    rf: PathBuf,
    original: &'static str,
}

const CUR: &str = "0.69.1-abc1234";
const RECIPE: &str = "schema: cookbook-recipe/v1\nid: r1\nargv:\n- apr\n- capability\n- --json\nexpect:\n  exit: 0\nhosts:\n- lambda\n- gx10\n";

impl Fixture {
    fn new() -> Self {
        let tmp = tempfile::tempdir().expect("tempdir");
        let rdir = tmp.path().join("recipes");
        let xdir = tmp.path().join("receipts");
        fs::create_dir_all(&rdir).expect("recipes dir");
        let rf = rdir.join("r1.yaml");
        fs::write(&rf, RECIPE).expect("recipe");
        Self {
            _tmp: tmp,
            rdir,
            xdir,
            rf,
            original: RECIPE,
        }
    }

    /// Fresh receipts: CURRENT set, both hosts PASS.
    fn reset(&self) {
        let _ = fs::remove_dir_all(&self.xdir);
        fs::create_dir_all(&self.xdir).expect("receipts dir");
        fs::write(self.xdir.join("CURRENT"), format!("{CUR}\n")).expect("CURRENT");
        write_receipt(&self.xdir, CUR, "lambda", &self.rf, &[]);
        write_receipt(&self.xdir, CUR, "gx10", &self.rf, &[]);
    }

    fn run(&self) -> i32 {
        check(&self.rdir, &self.xdir).0
    }
}

fn self_test() -> i32 {
    let fx = Fixture::new();
    let mut bad = 0;
    let mut row = |name: &str, want: i32, got: i32| {
        let ok = want == got;
        if !ok {
            bad += 1;
        }
        if ok {
            println!("ok    {name}");
        } else {
            println!("FAIL  {name} (rc {got}, wanted {want})");
        }
    };

    fx.reset();
    row("both hosts PASS from CURRENT -> PASS", 0, fx.run());
    fx.reset();
    fs::remove_file(fx.xdir.join(CUR).join("gx10").join("r1.json")).expect("unlink");
    row("MUST-RED: one host's receipt missing -> FAIL", 1, fx.run());
    fx.reset();
    write_receipt(
        &fx.xdir,
        CUR,
        "gx10",
        &fx.rf,
        &[
            ("verdict", "FAIL".into()),
            ("reasons", serde_json::json!(["stdout lacks '4'"])),
        ],
    );
    row("MUST-RED: a FAIL verdict -> FAIL", 1, fx.run());
    fx.reset();
    write_receipt(
        &fx.xdir,
        CUR,
        "lambda",
        &fx.rf,
        &[("verdict", "NOT_RUN".into())],
    );
    row("MUST-RED: NOT_RUN is not a pass -> FAIL", 1, fx.run());
    fx.reset();
    write_receipt(
        &fx.xdir,
        CUR,
        "lambda",
        &fx.rf,
        &[("apr_version", "apr 0.69.0 (0ld0ld0)".into())],
    );
    row(
        "MUST-RED: a receipt naming another binary -> FAIL",
        1,
        fx.run(),
    );
    fx.reset();
    fs::write(&fx.rf, format!("{}# edited\n", fx.original)).expect("edit");
    row(
        "MUST-RED: the recipe edited after its receipt -> FAIL (re-run required)",
        1,
        fx.run(),
    );
    fs::write(&fx.rf, fx.original).expect("restore");
    fx.reset();
    fs::write(fx.xdir.join("CURRENT"), "0.70.0-def5678\n").expect("CURRENT moved");
    row(
        "MUST-RED: CURRENT moved to a new release; old receipts do not count -> FAIL",
        1,
        fx.run(),
    );
    fx.reset();
    fs::remove_file(fx.xdir.join("CURRENT")).expect("unlink CURRENT");
    row("REFUSED: no CURRENT file -> rc 2", 2, fx.run());
    // Write the bad recipe FIRST, then the receipts, so both carry its sha: only the
    // non-string host check can turn this run red (a stale sha would make it red anyway).
    fs::write(
        &fx.rf,
        "schema: cookbook-recipe/v1\nid: r1\nhosts:\n- lambda\n- 2080\n",
    )
    .expect("bad hosts");
    fx.reset();
    row(
        "MUST-RED: a non-string host entry refuses the recipe -> FAIL",
        1,
        fx.run(),
    );
    fs::write(&fx.rf, fx.original).expect("restore");
    fx.reset();

    println!(
        "check_recipe_receipts self-test: {} ({bad} failed)",
        if bad == 0 { "PASS" } else { "FAIL" }
    );
    i32::from(bad != 0)
}

fn usage_exit() -> ! {
    eprintln!("usage: check_recipe_receipts [--recipes DIR] [--receipts DIR] | --self-test");
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut recipes = PathBuf::from("recipes/apr");
    let mut receipts = PathBuf::from("receipts");
    let mut test = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--self-test" => test = true,
            "--recipes" => recipes = PathBuf::from(it.next().unwrap_or_else(|| usage_exit())),
            "--receipts" => receipts = PathBuf::from(it.next().unwrap_or_else(|| usage_exit())),
            _ => usage_exit(),
        }
    }
    if test {
        std::process::exit(self_test());
    }
    let (rc, lines) = check(&recipes, &receipts);
    for l in lines {
        println!("{l}");
    }
    std::process::exit(rc);
}

#[cfg(test)]
mod tests {
    #[test]
    fn self_test_passes() {
        assert_eq!(super::self_test(), 0);
    }
}
