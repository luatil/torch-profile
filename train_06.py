"""
Main difference from train_05.py to train_06.py is increasing the batch_size

"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import (
    profile,
    schedule,
    tensorboard_trace_handler,
    ProfilerActivity,
)


# ---------- Model ---------------------------------------------------------- #


class TransformerBlock(nn.Module):
    """A single pre-norm transformer block. Small enough to train on one GPU."""

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class TinyLM(nn.Module):
    """Toy stack of transformer blocks with a projection head."""

    def __init__(self, vocab=8192, d_model=512, n_layers=4, seq_len=256):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.seq_len = seq_len

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


# ---------- Training loop with profiler ------------------------------------ #


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print(
            "WARNING: CUDA not available — trace will still work but GPU analysis is the point."
        )

    torch.manual_seed(0)

    # Deliberately small batch — keeps kernels launch-bound so the pattern shows.
    BATCH, SEQ, VOCAB = 32, 256, 8192

    model = TinyLM(vocab=VOCAB, seq_len=SEQ).to(device)
    # foreach=False → one kernel per parameter tensor. Good for demonstrating
    # "many tiny kernels" in the analyzer.
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)

    def make_batch():
        x = torch.randint(0, VOCAB, (BATCH, SEQ), device=device)
        y = torch.randint(0, VOCAB, (BATCH, SEQ), device=device)
        return x, y

    # Profiler schedule:
    #   wait=1     — skip 1 step (allocator warmup, first-call overhead)
    #   warmup=1   — run 1 step "inside" the profiler but discard
    #   active=5   — record 5 steps
    #   repeat=1   — do this cycle once
    # Total = 7 steps; trace covers the last 5.
    prof_schedule = schedule(wait=1, warmup=1, active=5, repeat=1)

    trace_dir = "./tb_trace"
    os.makedirs(trace_dir, exist_ok=True)

    model = torch.compile(model, mode="reduce-overhead")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        on_trace_ready=tensorboard_trace_handler(trace_dir),
        record_shapes=True,
        with_stack=False,  # stacks bloat the trace; enable for attribution work
        profile_memory=False,  # keep the trace small & readable
    ) as prof:
        for step in range(7):
            x, y = make_batch()

            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
            loss.backward()
            optim.step()

            # ---- Anti-pattern #1: .item() forces a sync ------------------- #
            # This is the classic culprit for unexpected GPU idle time.
            if step % 7 == 0:
                loss_val = loss.item()  # ← sync point

            prof.step()

    # Find the .json file the handler wrote and copy it to a stable name.
    files = sorted(os.listdir(trace_dir))
    trace_files = [f for f in files if f.endswith(".json") or f.endswith(".json.gz")]
    assert trace_files, f"No trace files found in {trace_dir}"
    latest = os.path.join(trace_dir, trace_files[-1])
    out = "./trace.json.gz" if latest.endswith(".gz") else "./trace.json"
    import shutil

    shutil.copyfile(latest, out)
    print(f"Wrote trace: {out}  (final loss ≈ {loss_val:.3f})")


if __name__ == "__main__":
    main()
