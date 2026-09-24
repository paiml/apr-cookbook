#!/usr/bin/env python3
"""Run cookbook-recipe/v1 recipes against ONE apr binary and write a receipt per recipe (#439).

Each `recipes/apr/**/<id>.yaml` (format: aprender#3769) names an argv, pinned models,
and an `expect`. This runner resolves each `{model:<slot>}` BY SHA256 from the host's model
directories, runs the argv, judges `expect`, and writes

    receipts/<apr-version>/<host>/<id>.json

A receipt is evidence only if it can name what produced it, so each one records the apr
`--version` line and binary sha256, the host, every model's path and sha256, the
resolved argv, rc, a digest and head of stdout, and the verdict with its reasons.

VERDICTS: PASS, FAIL (it ran and `expect` did not hold), NOT_RUN (a pinned model is not on
this host, or the binary was refused). NOT_RUN is not a pass. Any non-PASS makes the
run's exit code non-zero (doctrine: a capability is proven or RED, never a third state).

REFUSALS before anything runs (rc 2): `--expect-version SHA` that the binary's
`--version` line does not contain. That guards against running a stale apr and filing its
receipts as the release's.

GPU recipes (`backends` containing `cuda`) run each argv under `gpu-q` when it is on
PATH, the fleet GPU lock. The runner itself never takes the lock, so it is safe to call
from inside other tooling.

Usage:
  run_recipes.py --apr PATH --host NAME [--recipes DIR] [--receipts DIR]
                 [--expect-version SHA] [--only ID ...] [--model-dirs A:B] [--timeout S]
  run_recipes.py --self-test
Exit: 0 all PASS | 1 some FAIL or NOT_RUN | 2 refused / usage
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import yaml

DEFAULT_MODEL_DIRS = "~/models:~/.apr/models:~/.cache/apr/models"
HASH_CACHE = pathlib.Path(os.path.expanduser("~/.cache/apr-cookbook/sha256-cache.json"))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class Hashes:
    """sha256 by (path, size, mtime): a multi-GB model is hashed once, not once per recipe."""

    def __init__(self, cache_path):
        self.path = cache_path
        try:
            self.db = json.loads(cache_path.read_text())
        except (OSError, ValueError):
            self.db = {}

    def of(self, p):
        st = p.stat()
        key = "%s|%d|%d" % (p, st.st_size, st.st_mtime_ns)
        if key not in self.db:
            self.db[key] = sha256_file(p)
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self.db))
            except OSError:
                pass  # a cache that cannot be written only costs time
        return self.db[key]


def resolve_models(models, dirs, hashes):
    """{slot: path} for every pinned model, or (None, why). A file is matched by sha256,
    never by name: a same-named file with other bytes is a different model."""
    found = {}
    for slot, m in (models or {}).items():
        want = (m or {}).get("sha256")
        if not want or len(want) != 64:
            return None, "model slot %r pins no sha256" % slot
        hit = None
        for d in dirs:
            base = pathlib.Path(os.path.expanduser(d))
            if not base.is_dir():
                continue
            for p in sorted(base.iterdir()):
                if p.is_file() and p.suffix in (".gguf", ".apr", ".safetensors") and hashes.of(p) == want:
                    hit = p
                    break
            if hit:
                break
        if hit is None:
            return None, "no file with sha256 %s (slot %r) in %s" % (want[:16], slot, ":".join(dirs))
        found[slot] = str(hit)
    return found, None


def json_pointer(doc, ptr):
    cur = doc
    for part in ptr.lstrip("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


def last_json(text):
    """The last top-level JSON document in stdout (apr may log a line before it)."""
    dec, i, last = json.JSONDecoder(), 0, None
    while i < len(text):
        j = text.find("{", i)
        if j < 0:
            break
        try:
            last, end = dec.raw_decode(text, j)
            i = end
        except ValueError:
            i = j + 1
    return last


def judge(expect, rc, out):
    why = []
    if "exit" in expect and rc != expect["exit"]:
        why.append("exit %s, expected %s" % (rc, expect["exit"]))
    for s in expect.get("stdout_contains") or []:
        if s not in out:
            why.append("stdout lacks %r" % s)
    if expect.get("stdout_json") or expect.get("stdout_json_is_object"):
        doc = last_json(out)
        if expect.get("stdout_json_is_object") and not isinstance(doc, dict):
            why.append("stdout holds no JSON object")
        for ptr, want in (expect.get("stdout_json") or {}).items():
            try:
                got = json_pointer(doc, ptr)
            except (KeyError, IndexError, TypeError, ValueError):
                why.append("stdout JSON has no %s" % ptr)
                continue
            if got != want:
                why.append("stdout JSON %s = %r, expected %r" % (ptr, got, want))
    if not expect:
        why.append("recipe declares no expect -- nothing to judge")
    return why


def run_one(recipe, rfile, a, apr_line, apr_sha, hashes):
    rec = {"schema": "cookbook-receipt/v1", "recipe": recipe["id"], "recipe_sha256": sha256_file(rfile),
           "host": a.host, "apr_version": apr_line, "apr_sha256": apr_sha,
           "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    models, why = resolve_models(recipe.get("models"), a.model_dirs.split(":"), hashes)
    if models is None:
        rec.update(verdict="NOT_RUN", reasons=[why])
        return rec
    rec["models"] = {s: {"path": p, "sha256": recipe["models"][s]["sha256"]} for s, p in models.items()}
    argv = []
    for tok in recipe["argv"]:
        if tok.startswith("{model:") and tok.endswith("}"):
            tok = models[tok[len("{model:"):-1]]
        argv.append(tok)
    argv[0] = a.apr
    gpu = "cuda" in (recipe.get("backends") or [])
    if gpu and shutil.which("gpu-q"):
        argv = ["gpu-q"] + argv
        rec["gpu_lock"] = "gpu-q"
    rec["argv"] = argv
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=a.timeout, stdin=subprocess.DEVNULL)
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        rec.update(verdict="FAIL", reasons=["timed out after %ss" % a.timeout])
        return rec
    rec.update(rc=rc, stdout_sha256=hashlib.sha256(out.encode()).hexdigest(),
               stdout_head=out[:600], stderr_tail=err[-600:])
    why = judge(recipe.get("expect") or {}, rc, out)
    rec.update(verdict="FAIL" if why else "PASS", reasons=why)
    return rec


def version_token(line):
    # "apr 0.69.1 (d8a6df53a)" -> "0.69.1-d8a6df53a"
    parts = line.replace("(", " ").replace(")", " ").split()
    return "-".join(parts[1:3]) if len(parts) >= 3 else "unknown"


def run(a):
    try:
        apr_line = subprocess.run([a.apr, "--version"], capture_output=True, text=True, timeout=60).stdout.splitlines()[0]
    except (OSError, IndexError, subprocess.TimeoutExpired) as e:
        print("REFUSED: %s --version gave nothing usable (%s)" % (a.apr, e), file=sys.stderr)
        return 2
    if a.expect_version and a.expect_version not in apr_line:
        print("REFUSED: %s is %r, not the expected %s -- a receipt from the wrong binary is worse "
              "than none" % (a.apr, apr_line, a.expect_version), file=sys.stderr)
        return 2
    apr_sha = sha256_file(a.apr)
    files = sorted(pathlib.Path(a.recipes).rglob("*.yaml"))
    if not files:
        print("REFUSED: no recipes under %s -- zero recipes is broken wiring, not a pass" % a.recipes, file=sys.stderr)
        return 2
    outdir = pathlib.Path(a.receipts) / version_token(apr_line) / a.host
    outdir.mkdir(parents=True, exist_ok=True)
    hashes = Hashes(HASH_CACHE if not a.hash_cache else pathlib.Path(a.hash_cache))
    tally = {"PASS": 0, "FAIL": 0, "NOT_RUN": 0}
    for f in files:
        r = yaml.safe_load(f.read_text())
        if a.only and r["id"] not in a.only:
            continue
        rec = run_one(r, f, a, apr_line, apr_sha, hashes)
        (outdir / ("%s.json" % r["id"])).write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
        tally[rec["verdict"]] += 1
        print("%-7s %s%s" % (rec["verdict"], r["id"], ("  -- " + "; ".join(rec["reasons"])) if rec["reasons"] else ""))
    n = sum(tally.values())
    print("RESULT pass=%d fail=%d not_run=%d total=%d apr=%r host=%s receipts=%s"
          % (tally["PASS"], tally["FAIL"], tally["NOT_RUN"], n, apr_line, a.host, outdir))
    if n == 0:
        print("REFUSED: --only matched no recipe", file=sys.stderr)
        return 2
    return 0 if tally["PASS"] == n else 1


def self_test():
    """Case table: a stub apr answers '4' to anything; each row asserts a verdict or rc."""
    t = pathlib.Path(tempfile.mkdtemp(prefix="run-recipes-selftest-"))
    stub = t / "apr"
    stub.write_text('#!/usr/bin/env bash\ncase "$1" in --version) echo "apr 0.69.1 (abc1234)";; '
                    'capability) echo \'{"archs": ["qwen2"]}\';; *) echo "thinking...<answer>4</answer>";; esac\n')
    stub.chmod(0o755)
    models = t / "models"
    models.mkdir()
    (models / "tiny.gguf").write_bytes(b"not a real model, only its sha256 matters")
    msha = sha256_file(models / "tiny.gguf")
    rdir = t / "recipes"
    rdir.mkdir()

    def recipe(rid, expect, sha=msha, argv=None):
        (rdir / (rid + ".yaml")).write_text(yaml.safe_dump({
            "schema": "cookbook-recipe/v1", "id": rid,
            "argv": argv or ["apr", "run", "{model:main}", "--prompt", "What is 2+2?"],
            "models": {"main": {"role": "model", "sha256": sha}} if argv is None else {},
            "expect": expect, "min_apr": "0.69.1", "hosts": ["t"], "backends": ["cpu"]}))

    recipe("known-answer", {"exit": 0, "stdout_contains": ["<answer>4</answer>"]})
    recipe("wrong-answer", {"exit": 0, "stdout_contains": ["<answer>5</answer>"]})
    recipe("missing-model", {"exit": 0}, sha="0" * 64)
    recipe("json-ok", {"exit": 0, "stdout_json": {"/archs/0": "qwen2"}}, argv=["apr", "capability", "--json"])
    recipe("json-wrong", {"exit": 0, "stdout_json": {"/archs/0": "llama"}}, argv=["apr", "capability", "--json"])
    recipe("no-expect", {})

    def go(*extra):
        ns = argparse.Namespace(apr=str(stub), host="t", recipes=str(rdir), receipts=str(t / "receipts"),
                                expect_version="", only=[], model_dirs=str(models), timeout=30,
                                hash_cache=str(t / "hc.json"))
        for k, v in extra:
            setattr(ns, k, v)
        return run(ns)

    bad = 0

    def row(name, ok):
        nonlocal bad
        print(("ok    " if ok else "FAIL  ") + name)
        bad += 0 if ok else 1

    rc = go()
    rdir_out = t / "receipts" / "0.69.1-abc1234" / "t"
    verdict = lambda rid: json.loads((rdir_out / (rid + ".json")).read_text())["verdict"]
    row("rc 1 when any recipe is not PASS", rc == 1)
    row("CRUX known answer '4' -> PASS", verdict("known-answer") == "PASS")
    row("POSITIVE CONTROL: expecting '5' on the same answer -> FAIL", verdict("wrong-answer") == "FAIL")
    row("a pinned model absent from the host -> NOT_RUN, never PASS", verdict("missing-model") == "NOT_RUN")
    row("stdout_json pointer that holds -> PASS", verdict("json-ok") == "PASS")
    row("stdout_json pointer with another value -> FAIL", verdict("json-wrong") == "FAIL")
    row("a recipe with no expect -> FAIL (nothing judged is not a pass)", verdict("no-expect") == "FAIL")
    rec = json.loads((rdir_out / "known-answer.json").read_text())
    row("the receipt names the binary (version line + sha256) and the model sha256",
        rec["apr_version"] == "apr 0.69.1 (abc1234)" and len(rec["apr_sha256"]) == 64
        and rec["models"]["main"]["sha256"] == msha)
    row("--only the known-answer recipe -> rc 0", go(("only", ["known-answer"])) == 0)
    row("REFUSAL: --expect-version of another sha -> rc 2 before any recipe runs",
        go(("expect_version", "d8a6df53a")) == 2)
    row("REFUSAL: an empty recipes dir -> rc 2", go(("recipes", str(t / "none"))) == 2)
    shutil.rmtree(t, ignore_errors=True)
    print("self-test: %s (%d failed)" % ("PASS" if bad == 0 else "FAIL", bad))
    return 0 if bad == 0 else 1


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--apr")
    ap.add_argument("--host")
    ap.add_argument("--recipes", default="recipes/apr")
    ap.add_argument("--receipts", default="receipts")
    ap.add_argument("--expect-version", default="")
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--model-dirs", default=DEFAULT_MODEL_DIRS)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--hash-cache", default="")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.apr or not a.host:
        ap.error("--apr and --host are required")
    return run(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
