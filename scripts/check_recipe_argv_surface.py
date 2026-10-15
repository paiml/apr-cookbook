#!/usr/bin/env python3
"""Interim argv check for recipes/apr/**.yaml against ONE apr binary's --help (#3769 draft).

NOT the cookbook-recipe-v1 SHACL shape (aprender#3769 done_when 1, not built yet). It
checks one thing: every `--flag` a recipe's argv uses is listed by `apr <verb> --help`
on the given binary. A removed or misspelt flag is a failure. Nothing is run.
Usage: check_recipe_argv_surface.py <apr-binary> [recipes-dir]; exit 1 on any miss.
"""
import pathlib, re, subprocess, sys, yaml

apr, root = sys.argv[1], pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "recipes/apr")
bad = n = 0
for f in sorted(root.rglob("*.yaml")):
    r = yaml.safe_load(f.read_text())
    argv = r["argv"]
    assert argv[0] == "apr", f
    verb = argv[1]
    helptext = subprocess.run([apr, verb, "--help"], capture_output=True, text=True).stdout
    listed = set(re.findall(r"(--[a-z][a-z0-9-]*)", helptext))
    missing = [a for a in argv[2:] if a.startswith("--") and a not in listed]
    n += 1
    if missing:
        bad += 1
        print("FAIL %s: %s not in `apr %s --help`" % (r["id"], missing, verb))
    else:
        print("ok   %s" % r["id"])
ver = subprocess.run([apr, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
print("%d recipe(s), %d failed, against %s" % (n, bad, ver))
sys.exit(1 if bad or n == 0 else 0)
