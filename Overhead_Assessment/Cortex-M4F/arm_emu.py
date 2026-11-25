"""
Cycle-agnostic Thumb-2 / VFPv4 emulator for the Cortex-M4F kernels in
mcu/build/kernels.elf, with single (or multiple) instruction-skip injection.

Skip fault model (DeepLaser-style):
  * the targeted *dynamic* instruction is not executed (it behaves as a NOP):
    no register / flag / memory side effect, the PC falls through to the next
    sequential instruction (so a skipped branch is "not taken").
  * skipping an IT instruction removes the predication of the instruction(s)
    it guards (they execute unconditionally) - this is the real HW behaviour.
  * skipping an instruction inside an IT block is emulated as
    "execute-then-undo" so the IT state machine stays consistent.

The kernels are compiled with -mcpu=cortex-m4 (Thumb-2 + FPv4-SP); the same
encodings are executed here on Unicorn's ARMv7 core with VFP enabled.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np
from capstone import CS_ARCH_ARM, CS_MODE_THUMB, CS_MODE_MCLASS, Cs
from elftools.elf.elffile import ELFFile
from unicorn import (UC_ARCH_ARM, UC_HOOK_CODE, UC_HOOK_MEM_INVALID,
                     UC_MODE_THUMB, Uc, UcError)
from unicorn.arm_const import *  # noqa: F401,F403

CODE_BASE = 0x00010000
CODE_SIZE = 0x00010000
RET_ADDR = 0x0001F000          # inside code map; reaching it == kernel returned
DATA_BASE = 0x01000000
DATA_SIZE = 0x02000000         # 32 MB of arrays
STACK_TOP = 0x00F00000
STACK_SIZE = 0x00100000

R_REGS = [UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3,
          UC_ARM_REG_R4, UC_ARM_REG_R5, UC_ARM_REG_R6, UC_ARM_REG_R7,
          UC_ARM_REG_R8, UC_ARM_REG_R9, UC_ARM_REG_R10, UC_ARM_REG_R11,
          UC_ARM_REG_R12, UC_ARM_REG_SP, UC_ARM_REG_LR]
S_REGS = [getattr(__import__("unicorn.arm_const", fromlist=["x"]), f"UC_ARM_REG_S{i}")
          for i in range(32)]


@dataclass
class Insn:
    addr: int
    size: int
    mnemonic: str
    op_str: str
    in_it_body: bool = False

    @property
    def text(self):
        return f"{self.mnemonic} {self.op_str}".strip()


@dataclass
class RunResult:
    status: str                 # ok | crash | hang
    outputs: dict = field(default_factory=dict)
    n_dyn: int = 0              # dynamic instructions executed (if traced)
    trace: list | None = None   # list of addresses (if traced)
    err: str = ""
    ret: int = 0                # r0 on return (int-returning kernels)


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


class KernelEmu:
    def __init__(self, elf_path: str):
        with open(elf_path, "rb") as fh:
            elf = ELFFile(fh)
            text = elf.get_section_by_name(".text")
            self.text_addr = text["sh_addr"]
            self.text_bytes = text.data()
            self.syms = {}
            for s in elf.get_section_by_name(".symtab").iter_symbols():
                if s["st_info"]["type"] == "STT_FUNC" and s["st_size"] > 0:
                    self.syms[s.name] = (s["st_value"] & ~1, s["st_size"])
        self.cs = Cs(CS_ARCH_ARM, CS_MODE_THUMB | CS_MODE_MCLASS)
        self._uc = None
        self._hooks_dirty = False
        self.insns: dict[str, list[Insn]] = {}
        for name, (addr, size) in self.syms.items():
            self.insns[name] = self._disasm(addr, size)

    # ------------------------------------------------------------------ disasm
    def _disasm(self, addr, size):
        off = addr - self.text_addr
        code = self.text_bytes[off: off + size]
        out, it_left = [], 0
        for i in self.cs.disasm(code, addr):
            if i.mnemonic == ".word" or i.mnemonic.startswith(".") :
                break
            ins = Insn(i.address, i.size, i.mnemonic, i.op_str, in_it_body=it_left > 0)
            if it_left > 0:
                it_left -= 1
            if i.mnemonic in ("it", "itt", "ite", "ittt", "itte", "itet", "itee",
                              "itttt", "ittte", "ittet", "ittee", "itett", "itete", "iteet", "iteee"):
                # it, itt, ite, ittt ... -> number of guarded instructions
                it_left = len(i.mnemonic) - 1
            out.append(ins)
        # drop trailing literal-pool words disassembled as code after the last return
        return out

    def insn_at(self, func: str) -> dict[int, Insn]:
        return {i.addr: i for i in self.insns[func]}

    def all_insn_at(self) -> dict[int, Insn]:
        if not hasattr(self, "_all_iat"):
            self._all_iat = {i.addr: i for lst in self.insns.values() for i in lst}
        return self._all_iat

    # -------------------------------------------------------------------- core
    def _get_uc(self):
        """One Unicorn instance is reused; registers are zeroed on every run so
        that 'stale register' fault effects are deterministic (stale = 0)."""
        if self._uc is None:
            self._uc = self._new_uc()
        uc = self._uc
        for r in R_REGS:
            uc.reg_write(r, 0)
        for r in S_REGS:
            uc.reg_write(r, 0)
        uc.reg_write(UC_ARM_REG_FPSCR, 0)
        uc.reg_write(UC_ARM_REG_CPSR, uc.reg_read(UC_ARM_REG_CPSR) & 0x0F0F03FF | 0x20)  # clear NZCV/IT, thumb
        return uc

    def _new_uc(self):
        uc = Uc(UC_ARCH_ARM, UC_MODE_THUMB)
        uc.mem_map(CODE_BASE, CODE_SIZE)
        uc.mem_write(self.text_addr, bytes(self.text_bytes))
        uc.mem_write(RET_ADDR, b"\x00\xbf" * 8)          # nop sled at return address
        uc.mem_map(DATA_BASE, DATA_SIZE)
        uc.mem_map(STACK_TOP - STACK_SIZE, STACK_SIZE)
        # enable VFP: CPACR cp10/cp11 full access, FPEXC.EN
        cpacr = uc.reg_read(UC_ARM_REG_C1_C0_2)
        uc.reg_write(UC_ARM_REG_C1_C0_2, cpacr | (0xF << 20))
        uc.reg_write(UC_ARM_REG_FPEXC, 0x40000000)
        return uc

    def run(self, func: str, args: list, *, skips: list[tuple[int, int]] | None = None,
            skip_dyn: set[int] | None = None, trace: bool = False,
            target_func: str | None = None, target_calls: set[int] | None = None,
            max_insns: int = 50_000_000, out_fill: float = 0.0,
            trace_all: bool = False) -> RunResult:
        """
        func:   function called (a kernel or a *_rows driver).
        args:   list of ("in", np.ndarray) | ("out", n) | ("int", v) | ("float", v)
        skips:  list of (static_addr, occurrence) inside `target_func`
                (default: func). occurrence k>=0 -> skip the k-th execution of
                that instruction *within one call* of target_func;
                occurrence -1 -> skip every execution (persistent).
        target_calls: which calls (0-based entries) of target_func are faulted;
                None = every call.
        skip_dyn: set of dynamic-instruction indices (within func) to skip.
        out_fill: initial value of output buffers ("stale" memory content).
        trace_all: with trace=True, also trace callees (whole code region), not only `func`.
        """
        uc = self._get_uc()
        faddr, fsize = self.syms[func]
        tfunc = target_func or func
        taddr, _ = self.syms[tfunc]
        iat = {**self.all_insn_at(), **self.insn_at(tfunc)}   # skips may target callees

        # ---- marshal arguments (AAPCS hard-float)
        ptr = DATA_BASE
        int_args, flt_args, outputs = [], [], {}
        for k, a in enumerate(args):
            kind, val = a
            if kind == "in":
                arr = np.ascontiguousarray(val, dtype=np.float32)
                uc.mem_write(ptr, arr.tobytes())
                int_args.append(ptr)
                ptr += (arr.nbytes + 63) & ~63
            elif kind == "out":
                n = int(val)
                uc.mem_write(ptr, np.full(n, out_fill, np.float32).tobytes())
                outputs[k] = (ptr, n)
                int_args.append(ptr)
                ptr += (4 * n + 63) & ~63
            elif kind == "int":
                int_args.append(int(val) & 0xFFFFFFFF)
            elif kind == "float":
                flt_args.append(float(val))
        assert ptr < DATA_BASE + DATA_SIZE, "arguments exceed emulator data region"
        sp = STACK_TOP - 0x100
        stack_args = int_args[4:]
        sp -= 4 * len(stack_args)
        sp &= ~7
        for j, v in enumerate(stack_args):
            uc.mem_write(sp + 4 * j, struct.pack("<I", v))
        for j, v in enumerate(int_args[:4]):
            uc.reg_write(R_REGS[j], v)
        for j, v in enumerate(flt_args):
            uc.reg_write(S_REGS[j], _f32_bits(v))
        uc.reg_write(UC_ARM_REG_SP, sp)
        uc.reg_write(UC_ARM_REG_LR, RET_ADDR | 1)

        # ---- fault set-up
        skips = skips or []
        skip_dyn = skip_dyn or set()
        by_addr = {}
        for a, occ in skips:
            by_addr.setdefault(a, set()).add(occ)
        state = {"dyn": 0, "hits": {}, "undo": None, "call": -1,
                 "trace": [] if trace else None}
        per_call = tfunc != func or target_calls is not None

        # once every requested single skip has fired, stop, drop hooks, resume at JIT speed
        n_required = None
        if skips and not skip_dyn and not trace and all(o >= 0 for _, o in skips):
            if not per_call:
                n_required = len(skips)
            elif target_calls is not None:
                n_required = len(skips) * len(target_calls)

        def _fired(uc_):
            state["fired"] += 1
            if n_required is not None and state["fired"] >= n_required:
                state["stopped"] = True
                uc_.emu_stop()
                return True
            return False

        state["fired"] = 0
        state["stopped"] = False

        def entry_hook(uc_, address, size, _ud):
            state["call"] += 1
            state["hits"] = {}

        def hook(uc_, address, size, _ud):
            st = state
            if st["undo"] is not None:            # finish an execute-then-undo skip
                regs, sregs = st["undo"]
                for r, v in zip(R_REGS, regs):
                    uc_.reg_write(r, v)
                for r, v in zip(S_REGS, sregs):
                    uc_.reg_write(r, v)
                st["undo"] = None
                if _fired(uc_):
                    return
            if address == RET_ADDR:
                return
            if not (address in by_addr or skip_dyn or st["trace"] is not None):
                return                            # successor hook used only for undo
            d = st["dyn"]
            st["dyn"] = d + 1
            if st["trace"] is not None:
                st["trace"].append(address)
            do_skip = d in skip_dyn
            if not do_skip and address in by_addr:
                if target_calls is not None and st["call"] not in target_calls:
                    return
                h = st["hits"].get(address, 0)
                st["hits"][address] = h + 1
                occs = by_addr[address]
                do_skip = (h in occs) or (-1 in occs)
            if do_skip:
                ins = iat.get(address)
                if ins is not None and ins.in_it_body:
                    st["undo"] = ([uc_.reg_read(r) for r in R_REGS],
                                  [uc_.reg_read(r) for r in S_REGS])
                else:
                    uc_.reg_write(UC_ARM_REG_PC, (address + size) | 1)
                    _fired(uc_)

        handles, patched = [], []
        persistent_all = (skips and not skip_dyn and not trace and target_calls is None
                          and all(o == -1 for _, o in skips))
        if persistent_all:
            # every execution skipped == instruction replaced by a NOP (fast path)
            for a_ in by_addr:
                ins = iat[a_]
                orig = bytes(uc.mem_read(a_, ins.size))
                uc.mem_write(a_, b"\x00\xbf" if ins.size == 2 else b"\xaf\xf3\x00\x80")
                patched.append((a_, orig))
        elif trace or skip_dyn:
            handles.append(uc.hook_add(UC_HOOK_CODE, hook, begin=(CODE_BASE if trace_all else faddr),
                                       end=(CODE_BASE + CODE_SIZE - 1 if trace_all else faddr + fsize - 1)))
        elif skips:
            # cheap path: hook only the targeted instructions (+ successor for IT undo)
            if per_call:
                handles.append(uc.hook_add(UC_HOOK_CODE, entry_hook, begin=taddr, end=taddr))
            for a_ in by_addr:
                handles.append(uc.hook_add(UC_HOOK_CODE, hook, begin=a_, end=a_))
                ins = iat.get(a_)
                if ins is not None and ins.in_it_body:
                    handles.append(uc.hook_add(UC_HOOK_CODE, hook, begin=a_ + ins.size, end=a_ + ins.size))

        if handles or patched or self._hooks_dirty:
            uc.ctl_flush_tb()          # translated blocks cache code bytes and hook presence
        self._hooks_dirty = bool(handles or patched)
        res = RunResult(status="ok")
        start = faddr | 1
        try:
            while True:
                uc.emu_start(start, RET_ADDR, count=max_insns)
                pc = uc.reg_read(UC_ARM_REG_PC) & ~1
                if pc == RET_ADDR:
                    break
                if state["stopped"] and handles:
                    for h in handles:
                        uc.hook_del(h)
                    handles = []
                    uc.ctl_flush_tb()
                    state["stopped"] = False
                    start = pc | 1
                    continue
                res.status = "hang"
                break
        except UcError as e:
            res.status = "crash"
            res.err = str(e)
        for h in handles:
            uc.hook_del(h)
        for a_, orig in patched:
            uc.mem_write(a_, orig)
        for k, (p, n) in outputs.items():
            res.outputs[k] = np.frombuffer(bytes(uc.mem_read(p, 4 * n)), dtype=np.float32).copy()
        r0 = uc.reg_read(UC_ARM_REG_R0)
        res.ret = r0 - (1 << 32) if r0 & 0x80000000 else r0
        res.n_dyn = state["dyn"]
        res.trace = state["trace"]
        return res
