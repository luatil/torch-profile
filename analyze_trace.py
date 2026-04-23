"""
analyze_trace.py — Parse a torch.profiler Chrome trace and surface:

  1. Top-3 GPU idle gaps  — periods where the GPU had no work, ranked by duration.
                            For each gap, we identify the CPU-side op that was
                            running at the time (likely root cause: a sync,
                            a Python op, an allocator call, a dataloader, etc.)

  2. Top kernels by total time — ranked by cumulative duration across the trace,
                                 with call counts and mean time. This is the
                                 "where is my GPU actually spending its time"
                                 question.

Usage:
    python analyze_trace.py trace.json.gz
    python analyze_trace.py trace.json --gaps 5 --kernels 15

The Chrome trace format (spec: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU)
is a JSON object with a "traceEvents" array. Each event has at minimum:
    ph : phase  ('X' = complete event with duration, 'i' = instant, etc.)
    name : event name (op name, kernel name, ...)
    ts : timestamp in microseconds
    dur : duration in microseconds (for 'X' events)
    pid, tid : process/thread ids
    cat : category ('cpu_op', 'kernel', 'gpu_memcpy', 'cuda_runtime', ...)
    args : free-form metadata (correlation id, shapes, stream, device, ...)
"""

import argparse
import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import List


# ---------- Loading -------------------------------------------------------- #


def load_trace(path: str) -> dict:
    """Load a Chrome trace JSON (optionally gzipped)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


# ---------- Event classification ------------------------------------------ #

# Categories that indicate actual GPU work (kernels & memcpys).
# Everything else on the device timeline (e.g. markers) is ignored.
GPU_WORK_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}

# Categories that indicate CPU-side activity worth blaming for idle gaps.
CPU_CATEGORIES = {"cpu_op", "user_annotation", "cuda_runtime", "cuda_driver"}


def is_gpu_work(event: dict) -> bool:
    if event.get("ph") != "X":
        return False
    cat = event.get("cat", "")
    # Some PyTorch versions use 'Kernel' (capitalized), others 'kernel'.
    return cat.lower() in GPU_WORK_CATEGORIES


def is_cpu_op(event: dict) -> bool:
    if event.get("ph") != "X":
        return False
    return event.get("cat", "").lower() in CPU_CATEGORIES


# ---------- Idle gap detection -------------------------------------------- #


@dataclass
class Gap:
    start_us: float
    end_us: float
    dur_us: float
    cpu_ops_during: List[str]  # CPU ops active during the gap

    @property
    def dur_ms(self) -> float:
        return self.dur_us / 1000.0


def find_idle_gaps(events: List[dict], top_k: int = 3) -> List[Gap]:
    """
    Find intervals on the GPU timeline where no kernel was running.

    Strategy:
      1. Collect all GPU-work events, sort by start time.
      2. Merge overlapping intervals (kernels can overlap across streams).
      3. The "gap" between merged interval i and i+1 is GPU idle time.
      4. For each of the top-K largest gaps, scan CPU events whose interval
         overlaps the gap and collect their names — this is our root-cause hint.
    """
    gpu_events = [e for e in events if is_gpu_work(e)]
    if not gpu_events:
        return []

    # (start, end) intervals.
    intervals = sorted((e["ts"], e["ts"] + e.get("dur", 0)) for e in gpu_events)

    # Merge overlaps. On a single-GPU trace with one compute stream and one
    # copy stream, kernels on different streams can genuinely overlap — we
    # treat the GPU as busy whenever *any* stream has work.
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    # Gaps are between consecutive merged intervals.
    gaps_raw = []
    for (s1, e1), (s2, e2) in zip(merged, merged[1:]):
        gap_dur = s2 - e1
        if gap_dur > 0:
            gaps_raw.append((e1, s2, gap_dur))

    # Sort by duration (descending) and keep the top K.
    gaps_raw.sort(key=lambda g: -g[2])
    top_gaps = gaps_raw[:top_k]

    # Now attribute each gap to CPU activity that was live during it.
    cpu_events = [e for e in events if is_cpu_op(e)]

    result = []
    for gap_start, gap_end, gap_dur in top_gaps:
        during = []
        for e in cpu_events:
            s = e["ts"]
            d = e.get("dur", 0)
            # CPU op overlaps gap if it starts before gap ends and ends after gap starts.
            if s < gap_end and s + d > gap_start:
                during.append(e["name"])
        # De-duplicate while preserving order of first appearance.
        seen = set()
        uniq = []
        for n in during:
            if n not in seen:
                seen.add(n)
                uniq.append(n)
        result.append(Gap(gap_start, gap_end, gap_dur, uniq[:8]))

    return result


# ---------- Kernel aggregation -------------------------------------------- #


@dataclass
class KernelStats:
    name: str
    total_us: float
    calls: int

    @property
    def mean_us(self) -> float:
        return self.total_us / self.calls if self.calls else 0.0


def aggregate_kernels(events: List[dict]) -> List[KernelStats]:
    """Group GPU-work events by name; sum durations and counts."""
    totals = defaultdict(float)
    counts = defaultdict(int)
    for e in events:
        if not is_gpu_work(e):
            continue
        name = e.get("name", "<unknown>")
        totals[name] += e.get("dur", 0)
        counts[name] += 1

    stats = [KernelStats(n, totals[n], counts[n]) for n in totals]
    stats.sort(key=lambda k: -k.total_us)
    return stats


# ---------- Reporting ----------------------------------------------------- #


def shorten(name: str, width: int = 60) -> str:
    """Kernel names can be enormous (templated CUTLASS). Trim for display."""
    if len(name) <= width:
        return name
    return name[: width - 1] + "…"


def report(events: List[dict], n_gaps: int, n_kernels: int) -> None:
    # Trace-wide totals
    gpu_total = sum(e.get("dur", 0) for e in events if is_gpu_work(e))
    first_ts = min((e["ts"] for e in events if "ts" in e), default=0)
    last_end = max((e["ts"] + e.get("dur", 0) for e in events if "ts" in e), default=0)
    wall = last_end - first_ts
    utilization = (gpu_total / wall * 100) if wall else 0.0

    print("=" * 72)
    print("TRACE SUMMARY")
    print("=" * 72)
    print(f"Wall-clock span     : {wall / 1000:10.2f} ms")
    print(f"GPU-busy time       : {gpu_total / 1000:10.2f} ms")
    print(
        f"GPU utilization     : {utilization:10.1f} %   "
        f"({'BAD — look at gaps' if utilization < 80 else 'healthy'})"
    )
    print()

    # --- Idle gaps ------------------------------------------------------- #
    gaps = find_idle_gaps(events, top_k=n_gaps)
    print("=" * 72)
    print(f"TOP-{n_gaps} GPU IDLE GAPS")
    print("=" * 72)
    if not gaps:
        print("  No gaps found (or no GPU events in trace).")
    else:
        for i, g in enumerate(gaps, 1):
            print(
                f"\n  [{i}] {g.dur_ms:.3f} ms   @ t={g.start_us / 1000:.2f}..{g.end_us / 1000:.2f} ms"
            )
            if g.cpu_ops_during:
                print(f"      CPU activity during gap (likely root cause):")
                for op in g.cpu_ops_during:
                    print(f"        · {shorten(op, 66)}")
            else:
                print(
                    "      (no CPU activity recorded — could be cudaStreamSynchronize"
                    " or a dataloader waiting on I/O)"
                )
    print()

    # --- Kernel bottlenecks --------------------------------------------- #
    kernels = aggregate_kernels(events)
    print("=" * 72)
    print(f"TOP-{n_kernels} KERNELS BY TOTAL TIME")
    print("=" * 72)
    print(
        f"  {'rank':>4}  {'total ms':>10}  {'calls':>7}  {'mean µs':>10}  {'% gpu':>6}  kernel"
    )
    print(
        f"  {'-' * 4:>4}  {'-' * 10:>10}  {'-' * 7:>7}  {'-' * 10:>10}  {'-' * 6:>6}  {'-' * 60}"
    )
    for i, k in enumerate(kernels[:n_kernels], 1):
        pct = (k.total_us / gpu_total * 100) if gpu_total else 0.0
        print(
            f"  {i:>4}  {k.total_us / 1000:>10.3f}  {k.calls:>7}  "
            f"{k.mean_us:>10.2f}  {pct:>5.1f}%  {shorten(k.name)}"
        )
    print()

    # --- Quick diagnostic hints ----------------------------------------- #
    print("=" * 72)
    print("HINTS")
    print("=" * 72)
    hints = []
    if utilization < 70:
        hints.append(
            "GPU utilization is low — kernels exist but the GPU waits a lot. "
            "Look at the gaps above and their CPU ops."
        )
    # Launch-bound signal: lots of kernels with very small mean duration.
    small_kernels = [k for k in kernels if k.mean_us < 20 and k.calls > 10]
    if len(small_kernels) > 10:
        hints.append(
            f"{len(small_kernels)} kernels average <20µs — you may be launch-bound. "
            "Consider: bigger batch, torch.compile, foreach/fused optimizers, CUDA graphs."
        )
    # Sync hints
    sync_names = {
        "cudaStreamSynchronize",
        "cudaDeviceSynchronize",
        "cudaMemcpyAsync",
        "cudaMemcpy",
    }
    for g in gaps:
        if any(any(s in op for s in sync_names) for op in g.cpu_ops_during):
            hints.append(
                "At least one top gap overlaps a cuda sync — classic .item() / .cpu() / "
                "print(loss) pattern. Delay the sync or batch it."
            )
            break
    if not hints:
        hints.append(
            "No obvious red flags in the summary. Dig into per-kernel rooflines in NSight Compute."
        )
    for h in hints:
        print(f"  • {h}")
    print()


# ---------- Entry point --------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description="Analyze a torch.profiler Chrome trace.")
    ap.add_argument("trace", help="Path to trace.json or trace.json.gz")
    ap.add_argument("--gaps", type=int, default=3, help="Number of idle gaps to report")
    ap.add_argument(
        "--kernels", type=int, default=10, help="Number of top kernels to report"
    )
    args = ap.parse_args()

    data = load_trace(args.trace)
    events = data.get("traceEvents", data) if isinstance(data, dict) else data
    report(events, n_gaps=args.gaps, n_kernels=args.kernels)


if __name__ == "__main__":
    main()
