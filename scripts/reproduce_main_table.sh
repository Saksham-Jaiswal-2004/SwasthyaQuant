#!/usr/bin/env bash
# One command that regenerates every number in the report. If this script does not
# reproduce the table, the table is not a result. (Week 4, step 18.)
#
# Order is deliberate: the control is fitted BEFORE the quantum models, so nobody can
# claim it was sized after the fact to make the circuit look good. `compare` refuses to
# emit a table that has a quantum row and no control row, so a half-finished run fails
# loudly instead of producing a publishable-looking lie.
set -euo pipefail
make audit
make baselines
make control
make kernel
make vqc
make compare
echo "main table -> results/main_table.md"
