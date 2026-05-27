#!/bin/bash
# run all standalone test scripts
# usage: cd dokumen-cli && bash tests/scripts/run_all.sh

set -e
cd "$(dirname "$0")/../.."

echo "=========================================="
echo "  dokumen-cli standalone test suite"
echo "=========================================="
echo ""

TOTAL_PASS=0
TOTAL_FAIL=0

run_test() {
    echo "▶ $1"
    if python3 "$1"; then
        echo ""
    else
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        echo "  ⚠ FAILED"
        echo ""
    fi
    TOTAL_PASS=$((TOTAL_PASS + 1))
}

run_test tests/scripts/test_dokurouter.py
run_test tests/scripts/test_decomposed_judge.py
run_test tests/scripts/test_memory_system.py
run_test tests/scripts/test_eval_harness.py
run_test tests/scripts/test_consensus.py
run_test tests/scripts/test_calibration.py
run_test tests/scripts/test_skills.py
run_test tests/scripts/test_distill.py

echo "=========================================="
echo "  $TOTAL_PASS/$TOTAL_PASS test suites ran"
if [ $TOTAL_FAIL -gt 0 ]; then
    echo "  $TOTAL_FAIL FAILED"
    exit 1
else
    echo "  all passed"
    exit 0
fi
