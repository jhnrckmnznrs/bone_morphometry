#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo '[1/5] formatting'
cargo fmt --all -- --check

echo '[2/5] tests'
cargo test --all-targets --all-features --locked

echo '[3/5] clippy'
cargo clippy --all-targets --all-features --locked -- -D warnings

echo '[4/5] release build'
cargo build --release --locked

echo '[5/5] descriptor catalog'
./target/release/bone-morphometry list-descriptors

if [[ -n "${REFERENCE_CSV:-}" ]]; then
  echo '[optional] manuscript reference comparison'
  python3 validation/compare_reference_outputs.py "$REFERENCE_CSV"
else
  echo 'Reference comparison skipped.'
  echo 'Set REFERENCE_CSV=/path/to/manuscript_descriptors.csv to run it.'
fi
