"""Run the Cortex-M4F MAED benchmark (bench.c, same code and inputs as the Arduino UNO
sketch) in the emulator and estimate clock cycles from the executed-instruction trace with
the Cortex-M4 TRM instruction timings (zero-wait-state memory).

    ./build.sh && python run_bench.py
"""
import os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arm_emu import KernelEmu  # noqa: E402

P = 2   # pipeline refill after a taken branch (TRM: 1..3)


def nregs(op):
    m = re.search(r"\{([^}]*)\}", op)
    if not m:
        return 1
    n = 0
    for part in m.group(1).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            n += int(re.sub(r"\D", "", b)) - int(re.sub(r"\D", "", a)) + 1
        elif part:
            n += 1
    return n


def cost(ins, taken):
    base, op = ins.mnemonic.split(".")[0], ins.op_str
    if base in ("vdiv", "vsqrt"):
        return 14
    if base in ("vfma", "vfms", "vfnma", "vfnms", "vmla", "vmls", "vnmla", "vnmls"):
        return 3
    if base in ("vldr", "vstr"):
        return 2
    if base in ("vldmia", "vstmia", "vldmdb", "vstmdb", "vpush", "vpop", "vldm", "vstm"):
        return 1 + nregs(op)
    if base.startswith("v"):
        return 1
    if base in ("ldr", "ldrb", "ldrh", "ldrsb", "ldrsh", "str", "strb", "strh", "ldrd", "strd"):
        return 2
    if base in ("ldm", "ldmia", "stm", "stmia", "stmdb", "push"):
        return 1 + nregs(op)
    if base == "pop":
        return 1 + nregs(op) + (P if "pc" in op else 0)
    if base in ("bl", "blx", "bx"):
        return 1 + P
    if base.startswith("b") or base in ("cbz", "cbnz"):
        return 1 + P if taken else 1
    if base in ("sdiv", "udiv"):
        return 7
    return 1


E = KernelEmu(os.path.join(HERE, "bench.elf"))
ADDR = {i.addr: i for lst in E.insns.values() for i in lst}

src = open(os.path.join(HERE, "..", "Arduino", "main.ino")).read()   # same 500 test inputs
blk = src[src.index("test_inputs[SAMPLES] PROGMEM = {"):]
blk = blk[blk.index("{") + 1: blk.index("};")]
inputs = np.array([float(v) for v in re.findall(r"-?\d+\.\d+", blk)], np.float32)
assert inputs.size == 500, inputs.size
xbuf = np.array([inputs[(i * 37) % 500] for i in range(128 + 8)], np.float32)


def run(func, args):
    r = E.run(func, args, trace=True, trace_all=True, max_insns=200_000_000)
    assert r.status == "ok", (func, r.status)
    tr, c = r.trace, 0
    for k, a in enumerate(tr):
        ins = ADDR[a]
        nxt = tr[k + 1] if k + 1 < len(tr) else None
        c += cost(ins, nxt is not None and nxt != a + ins.size)
    return c, len(tr), r.ret, r.outputs


rows = []
for name, f in [("sigmoid", "bench_sigmoid"), ("error_sigmoid", "bench_error_sigmoid"),
                ("tanh", "bench_tanh"), ("error_tanh", "bench_error_tanh")]:
    c, n, ret, out = run(f, [("in", inputs), ("out", 500), ("int", 500)])
    rows.append((name, c, n, ret, out[1]))
for name, f in [("l2norm", "bench_l2norm"), ("error_l2norm", "bench_error_l2norm")]:
    c, n, ret, out = run(f, [("in", xbuf), ("out", 128), ("out", 500), ("int", 500)])
    rows.append((name, c, n, ret, out[2]))

# sanity: outputs vs. float64 reference
xc = np.clip(inputs.astype(np.float64), -3, 3)
ref = {"sigmoid": 1 / (1 + np.exp(-xc)), "tanh": np.tanh(xc)}
lines = ["Cortex-M4F (gcc -Os, FPv4-SP), 500 calls, same code/inputs as the ATmega328P sketch;",
         "cycles from the executed trace priced with Cortex-M4 TRM timings (0 wait states)",
         f"{'function':14s} {'cycles':>12s} {'instr':>10s} {'overhead':>9s} {'alarms':>7s} {'max|y-ref|':>11s}"]
base = {}
for name, c, n, ret, out in rows:
    key = name.replace("error_", "")
    err = np.max(np.abs(out - ref[key])) if key in ref else float("nan")
    ov = "" if not name.startswith("error_") else f"{100 * (c - base[key]) / base[key]:8.1f}%"
    if not name.startswith("error_"):
        base[key] = c
    lines.append(f"{name:14s} {c:12d} {n:10d} {ov:>9s} {ret:7d} {err:11.2e}")
txt = "\n".join(lines)
print(txt)
