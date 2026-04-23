# Profiler Trace Analyzer — Methodology

A working analyzer is only half the point. The other half is building the mental model for reading timelines, which is the single most transferable skill in ML systems performance work. What follows is a full tour: the trace format itself, the two questions the analyzer answers and why those are the right questions, the specific anti-patterns it catches, and how every concept here carries directly over to NSight Systems and NSight Compute.

---

## What a Chrome trace actually is

`torch.profiler` emits a JSON file in the Chrome Trace Event format — the same format `chrome://tracing` and `ui.perfetto.dev` render. Strip away the wrapper and it's a flat list of events:

```json
{"ph": "X", "cat": "kernel", "name": "ampere_sgemm_128x64",
 "ts": 18734221, "dur": 127, "pid": 0, "tid": 7, "args": {...}}
```

The fields that matter for us:

- `ph` — phase. `"X"` is a "complete event" with a duration; these are the only ones we care about. `"i"` is an instant marker, `"s"/"f"` are flow-arrow endpoints (used to connect a CPU launch to its GPU kernel).
- `cat` — category. `"cpu_op"` is an ATen op called from Python. `"kernel"` is GPU work. `"cuda_runtime"` is a `cudaLaunchKernel` / `cudaMemcpyAsync` / `cudaStreamSynchronize` call. `"gpu_memcpy"` is a H2D/D2H transfer.
- `ts`, `dur` — microsecond timestamps, monotonic within the trace.
- `pid`, `tid` — the profiler uses fake PIDs/TIDs to put CPU and GPU on separate timeline tracks when you open the trace in Perfetto.

That is the entire ontology. Everything the analyzer does is: filter by `cat`, use `ts`+`dur` to compute intervals, and look at overlaps.

A practical habit: open a trace in **`ui.perfetto.dev`** (Perfetto superseded Chrome's internal viewer and is far better) while running the analyzer. The script tells you *what* is wrong; the visual tells you *where* and lets you zoom in on the neighborhood.

---

## The two questions that matter

Every performance investigation in training collapses to one of two questions:

1. **"Is my GPU idle, and if so, why?"** — a utilization problem. The GPU can execute trillions of FLOPs per second and you're not feeding it enough work. The fix is usually on the CPU, the data path, or the distribution strategy.

2. **"If my GPU is busy, is it busy on the right things?"** — a kernel efficiency problem. You've fed it work, but the specific kernels it's running are slow relative to what's achievable. The fix is algorithmic (kernel choice, fusion, precision) or microarchitectural (memory access patterns, occupancy).

You must answer question 1 first. There is no point optimizing a matmul kernel that only runs 40% of the time — you'd be tuning the engine while the car is parked. The analyzer answers both, in order.

---

## Question 1: GPU idle gaps

### The algorithm

1. Take every event with `cat == "kernel"` or `"gpu_memcpy"` — these are the only things that make the GPU busy.
2. Sort them by start time, then **merge overlapping intervals**. Merging matters because modern training uses multiple CUDA streams (typically one for compute, one for NCCL communication, one for copies). Two streams running simultaneously means the GPU is busy on *both*, so their intervals should coalesce into a single busy period.
3. Each gap between merged intervals is GPU idle time. Sort by duration, take top-K.
4. For each top gap, sweep CPU events whose interval overlaps the gap. The CPU ops live during that gap are the prime suspects.

The merging step is the one most people get wrong on their first try — without it, a trace with overlapping compute+comm streams shows phantom "gaps" that are actually busy-on-the-other-stream.

### What the CPU ops during a gap are telling you

When the GPU is idle and the CPU is busy, the thing the CPU is doing is — with very high probability — the cause. Four archetypes cover almost every case:

| CPU activity during gap | What it means | Fix |
|---|---|---|
| `cudaStreamSynchronize`, `cudaDeviceSynchronize` | Someone forced a blocking wait on the GPU. Usually `.item()`, `.cpu()`, `.numpy()`, or `print(tensor)`. | Defer the sync (log every N steps), accumulate on device, use `.detach()` instead of `.item()` where possible. |
| `DataLoader`, `enumerate`, `__next__` | Host-side data pipeline can't keep up. GPU finished step N and is waiting for batch N+1. | More `num_workers`, pinned memory + `non_blocking=True`, prefetch, NVIDIA DALI, or a faster storage backend. |
| Long `aten::*` with no kernel below it | An op that fell back to CPU (often happens with unusual dtypes, custom ops, or rare shape combos). | Find the op, force it to GPU, or replace with a GPU-native alternative. |
| `cudaMalloc`, `cudaFree` | Allocator churn. The caching allocator missed and went to the driver. | `torch.cuda.set_per_process_memory_fraction`, avoid varying shapes, check for memory fragmentation. |

### The "no CPU activity" case

Sometimes a gap has nothing on the CPU either. This is the hardest case. Possibilities: (a) you're waiting on NCCL (communication on another stream — if this is a distributed trace, look for `ncclKernel_AllReduce` stuck in its own gap with no overlapping compute), (b) the sync is happening inside a driver thread the profiler didn't capture, or (c) the profiler's `wait`/`warmup` boundary happened mid-step and this "gap" is artificial. Re-run with a longer `active` window before concluding.

---

## Question 2: Kernel bottlenecks

### The algorithm

Group all `cat == "kernel"` events by `name`, sum `dur`, count occurrences, sort by total duration. The analyzer prints total ms, call count, mean µs, and % of total GPU time.

### Reading the output

Three patterns to recognize:

**Pattern A — one kernel dominates (>40% of GPU time).** This is what you want. Optimization effort has a clear target. If it's a matmul (`ampere_gemm_*`, `cutlass::*`, `sm90_xmma_*`), move to NSight Compute and check the roofline: is it memory-bound or compute-bound? If compute-bound at high Tensor Core utilization, you're near the ceiling. If memory-bound, look at tile sizes, fusion opportunities, or whether the shape is just awkward for the hardware (e.g., dimensions not divisible by 8 kill Tensor Core throughput on FP16).

**Pattern B — the top is flat and populated by tiny kernels.** Dozens of kernels, each <20µs mean, each called hundreds of times. Total GPU time is dominated by kernel-launch overhead, not computation. The analyzer flags this as "launch-bound." Fixes in order of effort:

1. Larger batch size — more work per kernel amortizes the ~5µs launch cost.
2. `torch.compile(model)` — fuses pointwise ops aggressively; 5 tiny kernels become 1 bigger one.
3. Fused/`foreach` optimizer — `torch.optim.AdamW(..., fused=True)` turns N-parameter updates (N small kernels) into 1 kernel.
4. CUDA Graphs — captures a sequence of launches and replays them with one API call. Only works when shapes are static.

**Pattern C — memory ops are disproportionately large.** If `aten::copy_`, `Memcpy HtoD`, or `Memcpy DtoH` shows up in the top 5, you have a data transfer problem. D2H in particular usually means a hidden sync. H2D means your data loader is sending the batch *during* the step instead of ahead of it.

### Kernel names — a decoder ring

PyTorch kernels come from several sources and their names tell you which:

- `aten::*` — the PyTorch operator layer. This is a CPU-side label; the actual GPU kernel(s) are below it. Not usually shown in the kernel section of the analyzer unless `record_shapes` collapsed the view.
- `void at::native::*` — PyTorch's hand-written CUDA kernels (elementwise, reductions).
- `ampere_*`, `hopper_*`, `sm80_*`, `sm90_*` — cuBLAS. Architecture-specific GEMM kernels.
- `cutlass::*`, `sm90_xmma_*` — CUTLASS templates. Often appear when `torch.compile` codegens GEMMs or when using Flash Attention.
- `ncclKernel_*` — NCCL collectives. `AllReduce`, `Broadcast`, `AllGather` variants.
- `triton_*` or names with a hash suffix — Triton-compiled kernels (from `torch.compile`'s Inductor backend).

The name is the first clue to whether a kernel is something you can influence directly (Triton, custom CUDA) or something you have to coax into firing well (cuBLAS, cuDNN).

---

## The three anti-patterns in `train.py`

The training script deliberately includes three inefficiencies so the analyzer has something to surface:

**1. `loss.item()` inside the loop.** This forces a host-device sync. Every step, Python waits until the GPU has finished everything queued up to that point, then copies one scalar back. You'll see this in the trace as a large gap immediately after the backward pass, with `cudaStreamSynchronize` and `aten::item` on the CPU side during the gap.

**2. `BATCH=4`.** Modern GPUs are designed for a large number of parallel threads. A tiny batch means each kernel is short, the launch overhead (~5µs per kernel, fixed cost) is no longer amortized, and you become launch-bound. The fix is usually just "make the batch bigger," but sometimes you can't (memory limits, latency requirements), in which case `torch.compile` and CUDA graphs are the escape hatch.

**3. `foreach=False` on AdamW.** By default (since PyTorch 1.13) AdamW groups parameter updates. With `foreach=False` each parameter tensor becomes its own set of small kernels — for a model with 100 parameter tensors you get 100× the kernel launches on the optimizer step. With `foreach=True` (the default) it becomes a handful of large kernels; with `fused=True` it becomes literally one.

When you swap these three back (`fused=True`, batch ≥ 32, drop the `.item()`), you should see GPU utilization jump from ~30-50% into the 80s or 90s, the top gap shrink dramatically, and the kernel histogram consolidate onto a few large matmuls.

---

## How to run it

```bash
# One-time setup
pip install torch

# Capture a trace
python train.py
# → writes ./trace.json.gz

# Analyze
python analyze_trace.py trace.json.gz
python analyze_trace.py trace.json.gz --gaps 5 --kernels 20

# Visualize (optional but recommended — drag the .json.gz into)
#   https://ui.perfetto.dev
```

The training run takes about 10–30 seconds on a modern GPU. The analyzer runs in well under a second even on large traces.

---

## How this maps to NSight

`torch.profiler` is the right tool for the first question ("where is my framework spending time?"). NSight is the right tool for the next two questions. The handoff is deliberate.

### NSight Systems (`nsys`)

NSight Systems is the direct upgrade path from `torch.profiler`. Same timeline mental model, but:

- It sees the full OS picture: kernel launches, NCCL calls, CUDA API calls, OS thread scheduling, CPU sampling, NVTX ranges, and — critically — multi-process / multi-GPU coordination.
- It captures NCCL activity in full, so in a distributed run you can see "AllReduce on GPU 0 finished at t=X but GPU 3's AllReduce didn't start until t=X+2ms" — the straggler pattern that's invisible in a single-process torch trace.
- It runs at the process level, not inside Python, so it doesn't perturb the run the way `torch.profiler` can.

Everything in this analyzer transfers: `nsys` profiles are also opened in a timeline viewer, also have CPU and GPU tracks, also want you to find idle gaps and attribute them. The difference is you annotate regions with `torch.cuda.nvtx.range_push("forward")` / `range_pop()` (or `nvtx.annotate`) so your code's logical structure shows up on the timeline next to the low-level events.

Typical workflow:

```bash
nsys profile -o my_run --trace=cuda,nvtx,nccl,cudnn,cublas python train.py
nsys-ui my_run.nsys-rep
```

### NSight Compute (`ncu`)

Once you've identified a specific kernel that's eating 40% of your GPU time, NSight Compute tells you *why*. It runs that kernel in replay mode (same inputs, many times with different hardware counters enabled) and gives you:

- **Roofline analysis** — am I memory-bound or compute-bound? The roofline plot is the single most important diagnostic in kernel optimization. If you're on the memory roof, FLOPS optimization is a waste of time; you need to reduce data movement. If you're on the compute roof, you're near the ceiling and further gains are small.
- **Warp stall reasons** — when a warp isn't making progress, the SM records why (waiting on memory? barrier? pipeline stall? dependency?). The distribution over stall reasons tells you the bottleneck.
- **Memory throughput** — L1/L2/DRAM bandwidth used vs. theoretical max. Tensor Core utilization as a percentage of peak.
- **Occupancy** — how many warps are actually resident on the SM vs. the theoretical max, and what's limiting it (registers? shared memory? block size?).

`ncu` is slow (seconds to minutes per kernel because of replay) and narrow (one kernel at a time), which is why you use `nsys` or `torch.profiler` first to pick the target. The progression is always:

1. `torch.profiler` / `nsys` → find the hot kernel. *"Where is time going?"*
2. `ncu --kernel-name <name>` → find out why it's slow. *"Why is this kernel not at roofline?"*
3. Rewrite (Triton, CUTLASS, CUDA) → verify with `ncu` again.

### The analyzer as a pedagogical bridge

This analyzer is deliberately a toy: it does what NSight Systems does automatically (idle detection, top-kernel ranking) but from first principles over a JSON file. Writing it forces you to confront the fact that the "gaps" and "bottlenecks" NSight highlights are not magic — they're simple interval arithmetic over timestamped events. Once you've written this, the NSight UI stops being a mystery and starts being a faster, more featured version of the same thing. The transfer is almost one-to-one.

---

## Things to try next

A few small extensions, ranked roughly by effort-to-insight ratio:

1. **Add NVTX ranges to `train.py`** — wrap forward / backward / optimizer step in `torch.cuda.nvtx.range`. Re-run and in Perfetto you'll see labeled regions in the trace. The analyzer could be extended to aggregate time per NVTX range.
2. **Distinguish forward vs. backward kernels** — look at the `External id` / correlation id in `args` to link GPU kernels back to their CPU launcher, then group kernels by which high-level op launched them.
3. **Identify the critical path across streams** — when compute and NCCL run on different streams, the "real" step time is the max across streams, not the sum. Upgrade the analyzer to compute per-stream utilization separately.
4. **Flag H2D/D2H imbalance** — a healthy step has data flowing in (H2D) at the start and ideally zero D2H until logging. Count each and flag suspicious ratios.
5. **Compare two traces** — fix the `.item()` call, capture a second trace, and have the analyzer diff them ("utilization +41%, top-1 gap eliminated, optimizer time -82%"). This is the muscle you'll use constantly in real optimization work.

The last one is the most important habit. A trace in isolation tells you the current state; two traces tell you whether your change helped and by how much. All real performance work is diffs.
