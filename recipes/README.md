# `apr` binary recipes (`cookbook-recipe/v1`)

Each YAML file here is one command you can run against a released `apr` binary: the argv,
the model it needs (pinned by sha256), what a correct run prints, and the minimum apr
version. The format is the one decided in
[aprender#3769](https://github.com/paiml/aprender/issues/3769).

**What is validated today, and what is not.**

| Check | Status |
|---|---|
| Every `--flag` in a recipe's argv is listed by `apr <verb> --help` on the release binary | **Checked** by `scripts/check_recipe_argv_surface.py`: 5/5 against `apr 0.69.1 (d8a6df53a)`. A misspelt flag fails, and that negative control was run |
| The `cookbook-recipe-v1` SHACL shape (argv against the CLI surface JSON, receipt present, a recipe for every model-taking verb) | **Not built yet**: aprender#3769 done_when 1 |
| A receipt from the release binary on lambda and gx10 showing `expect` passed | **Runner built** (`scripts/run_recipes.py`, #439); receipts land under `receipts/<apr-version>/<host>/` once the post-release run executes on the published binary |

So these recipes are a draft. They are not SHACL-validated recipes yet, and the table
above says so instead of implying otherwise.

```bash
python3 scripts/check_recipe_argv_surface.py /path/to/apr
```

## Running them: `scripts/run_recipes.py` (#439)

```bash
python3 scripts/run_recipes.py --apr /path/to/apr --host lambda --expect-version <sha>
python3 scripts/run_recipes.py --self-test
```

- Each `{model:<slot>}` is resolved **by sha256** from `~/models`, `~/.apr/models` and `~/.cache/apr/models`. A file with the right name and different bytes is not the model.
- `--expect-version` refuses a binary whose `--version` line lacks that sha, before any recipe runs.
- GPU recipes run under `gpu-q` (the fleet GPU lock) when it is on PATH.
- Each run writes `receipts/<version>-<sha>/<host>/<id>.json`, holding the version line, binary sha256, model sha256s, resolved argv, rc, stdout digest and verdict.
- The verdict is `PASS`, `FAIL` or `NOT_RUN`. `NOT_RUN` means a pinned model is not on the host, and it is not a pass: any non-PASS makes the exit code non-zero.
- The self-test is an 11-row case table against a stub `apr`. Its CRUX row is the known answer `<answer>4</answer>` judged PASS; its positive control is the same answer expected as `5`, judged FAIL. A planted mutant that ignores `stdout_contains` turns the self-test red.

