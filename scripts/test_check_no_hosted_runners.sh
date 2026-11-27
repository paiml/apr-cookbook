#!/usr/bin/env bash
# test_check_no_hosted_runners.sh -- runs every fixture under tests/fixtures/hosted-runners
# through the gate and compares its exit code with the fixture's `expect` file.
# Also fails while the gate still calls an interpreter outside bash (python-ban, #470).
#
# Each fixture is a directory: wf/ (the workflow files), baseline (the ratchet rows) and
# expect (the exit code the gate on main gives for it).
#
# Usage: scripts/test_check_no_hosted_runners.sh   (env GATE overrides the gate path)
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
GATE=${GATE:-$ROOT/scripts/check_no_hosted_runners.sh}
FIX=$ROOT/tests/fixtures/hosted-runners
n=0
bad=0
for d in "$FIX"/*/; do
    name=$(basename "$d")
    n=$((n + 1))
    want=$(cat "$d/expect")
    rc=0
    WF_DIR="$d/wf" HOSTED_BASELINE="$d/baseline" bash "$GATE" > /dev/null 2>&1 || rc=$?
    if [ "$rc" = "$want" ]; then
        echo "ok    $name (rc $rc)"
    else
        echo "FAIL  $name (rc $rc, wanted $want)"
        bad=$((bad + 1))
    fi
done
if grep -q 'python' "$GATE"; then
    echo "FAIL  $GATE still calls python (python-ban, #470)"
    bad=$((bad + 1))
else
    echo "ok    $GATE is python-free"
fi
echo "check_no_hosted_runners fixtures: $n cases, $bad failed"
[ "$bad" = 0 ]
