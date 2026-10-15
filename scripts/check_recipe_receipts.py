#!/usr/bin/env python3
"""Required check: every recipe carries a PASS receipt from the CURRENT apr release (#442).

A receipt nobody reads is an artifact, not evidence. This check reads the committed
receipts that `scripts/run_recipes.py` writes (#439) and fails unless, for EVERY recipe
under recipes/apr/ and EVERY host that recipe declares:

  receipts/<CURRENT>/<host>/<id>.json exists,
  its verdict is PASS (FAIL and NOT_RUN are not passes),
  its apr_version names the CURRENT release's binary, and
  its recipe_sha256 equals the recipe file's sha256 now. An EDITED recipe invalidates its
  old receipt; otherwise a receipt could vouch for a command it never ran.

<CURRENT> is the one line in receipts/CURRENT, e.g. `0.69.1-d8a6df53a` (the directory
token run_recipes.py writes). The release step updates it (#444). A receipt from any other
release is ignored, so a stale release can never satisfy the check.

It never runs a model: CI holds none. It judges committed evidence only.

Usage: check_recipe_receipts.py [--recipes DIR] [--receipts DIR] | --self-test
Exit: 0 every recipe covered | 1 a gap or a bad receipt | 2 usage / no CURRENT / no recipes
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import tempfile

import yaml


def sha256_file(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def check(recipes_dir, receipts_dir):
    cur_file = pathlib.Path(receipts_dir) / "CURRENT"
    try:
        current = cur_file.read_text().strip()
    except OSError:
        print("REFUSED: no %s -- the check cannot know which release's receipts are required" % cur_file)
        return 2
    if not current or "-" not in current:
        print("REFUSED: %s holds %r, not a <version>-<sha> token" % (cur_file, current))
        return 2
    sha = current.split("-", 1)[1]
    files = sorted(pathlib.Path(recipes_dir).rglob("*.yaml"))
    if not files:
        print("REFUSED: no recipes under %s -- zero recipes is broken wiring, not a pass" % recipes_dir)
        return 2
    bad, checked = [], 0
    for f in files:
        r = yaml.safe_load(f.read_text())
        rid, hosts = r["id"], r.get("hosts") or []
        if not hosts:
            bad.append("%s: declares no hosts -- nothing could ever vouch for it" % rid)
            continue
        want_sha = sha256_file(f)
        for h in hosts:
            checked += 1
            rp = pathlib.Path(receipts_dir) / current / h / ("%s.json" % rid)
            if not rp.is_file():
                bad.append("%s @ %s: no receipt from %s (%s)" % (rid, h, current, rp))
                continue
            try:
                rec = json.loads(rp.read_text())
            except ValueError as e:
                bad.append("%s @ %s: receipt is not JSON (%s)" % (rid, h, e))
                continue
            if rec.get("verdict") != "PASS":
                bad.append("%s @ %s: verdict %s -- %s" % (rid, h, rec.get("verdict"), "; ".join(rec.get("reasons") or [])))
            if sha not in (rec.get("apr_version") or ""):
                bad.append("%s @ %s: receipt names %r, not the CURRENT binary %s" % (rid, h, rec.get("apr_version"), sha))
            if rec.get("recipe_sha256") != want_sha:
                bad.append("%s @ %s: the recipe changed since this receipt (receipt %s, file %s) -- re-run it"
                           % (rid, h, (rec.get("recipe_sha256") or "none")[:12], want_sha[:12]))
    for b in bad:
        print("FAIL  " + b)
    print("%s  %d recipe x host cell(s) checked against %s, %d problem(s)"
          % ("PASS" if not bad else "FAIL", checked, current, len(bad)))
    return 0 if not bad else 1


def self_test():
    t = pathlib.Path(tempfile.mkdtemp(prefix="check-receipts-selftest-"))
    rdir, xdir = t / "recipes", t / "receipts"
    rdir.mkdir()
    rf = rdir / "r1.yaml"
    rf.write_text(yaml.safe_dump({"schema": "cookbook-recipe/v1", "id": "r1", "argv": ["apr", "capability", "--json"],
                                  "expect": {"exit": 0}, "hosts": ["lambda", "gx10"]}))
    cur = "0.69.1-abc1234"

    def receipt(host, **over):
        d = xdir / cur / host
        d.mkdir(parents=True, exist_ok=True)
        rec = {"recipe": "r1", "verdict": "PASS", "reasons": [], "apr_version": "apr 0.69.1 (abc1234)",
               "recipe_sha256": sha256_file(rf)}
        rec.update(over)
        (d / "r1.json").write_text(json.dumps(rec))

    def reset():
        if xdir.exists():
            shutil.rmtree(xdir)
        xdir.mkdir()
        (xdir / "CURRENT").write_text(cur + "\n")
        receipt("lambda")
        receipt("gx10")

    bad = 0

    def row(name, want, got):
        nonlocal bad
        ok = want == got
        bad += 0 if ok else 1
        print(("ok    " if ok else "FAIL  ") + name + ("" if ok else " (rc %s, wanted %s)" % (got, want)))

    def run():
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            return check(rdir, xdir)

    reset(); row("both hosts PASS from CURRENT -> PASS", 0, run())
    reset(); (xdir / cur / "gx10" / "r1.json").unlink()
    row("MUST-RED: one host's receipt missing -> FAIL", 1, run())
    reset(); receipt("gx10", verdict="FAIL", reasons=["stdout lacks '4'"])
    row("MUST-RED: a FAIL verdict -> FAIL", 1, run())
    reset(); receipt("lambda", verdict="NOT_RUN")
    row("MUST-RED: NOT_RUN is not a pass -> FAIL", 1, run())
    reset(); receipt("lambda", apr_version="apr 0.69.0 (0ld0ld0)")
    row("MUST-RED: a receipt naming another binary -> FAIL", 1, run())
    reset(); rf.write_text(rf.read_text() + "# edited\n")
    row("MUST-RED: the recipe edited after its receipt -> FAIL (re-run required)", 1, run())
    rf.write_text(rf.read_text().replace("# edited\n", ""))
    reset(); (xdir / "CURRENT").write_text("0.70.0-def5678\n")
    row("MUST-RED: CURRENT moved to a new release; old receipts do not count -> FAIL", 1, run())
    reset(); (xdir / "CURRENT").unlink()
    row("REFUSED: no CURRENT file -> rc 2", 2, run())
    shutil.rmtree(t, ignore_errors=True)
    print("check_recipe_receipts self-test: %s (%d failed)" % ("PASS" if bad == 0 else "FAIL", bad))
    return 0 if bad == 0 else 1


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", default="recipes/apr")
    ap.add_argument("--receipts", default="receipts")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    return self_test() if a.self_test else check(a.recipes, a.receipts)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
