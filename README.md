# Profiler Trace Analyzer

Small project for the ML Systems curriculum: train a model, capture a
`torch.profiler` trace, and analyze it for GPU idle gaps and kernel bottlenecks
from first principles (no NSight, no TensorBoard plugin, just JSON + Python).

## Files

- **`train.py`** — Small transformer LM, trained for 7 steps with profiling
  active on the last 5. Deliberately contains three anti-patterns (`.item()`
  sync, tiny batch, non-fused optimizer) so the analyzer has something to find.
- **`analyze_trace.py`** — Parses the Chrome trace JSON and reports:
  - Trace summary with GPU utilization %
  - Top-3 GPU idle gaps with CPU-side root causes
  - Top-10 kernels ranked by total time, calls, mean duration, % of GPU time
  - Diagnostic hints based on the patterns observed
- **`METHODOLOGY.md`** — Full walkthrough: trace format, algorithm rationale,
  kernel-name decoder ring, anti-pattern guide, and the NSight handoff.

## Run

```bash
pip install torch
python train.py                      # writes ./trace.json.gz
python analyze_trace.py trace.json.gz
```

For visualization, drag the trace into https://ui.perfetto.dev.

## Exercise

After you've run it once and read the output, fix one anti-pattern at a time in
`train.py` and re-run. Watch GPU utilization climb from ~30–50% toward 90%, top
gaps shrink, and the kernel histogram consolidate onto a few large GEMMs.

The order to try:
1. Remove `loss.item()` (or move it outside the profiled steps). Biggest gap
   should disappear.
2. Bump `BATCH` from 4 to 32 or 64. Kernels become less launch-bound.
3. Change `foreach=False` → `fused=True` on the optimizer. Optimizer step
   collapses into a single kernel.

Then capture a second trace and compare. All real performance work is diffs.
