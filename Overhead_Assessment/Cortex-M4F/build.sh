#!/usr/bin/env bash
# Build taylor_kernels.c (Algorithm-5 kernels + benchmark) for the Cortex-M4F (needs the GNU Arm Embedded toolchain, arm-none-eabi-gcc).
set -euo pipefail
cd "$(dirname "$0")"
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -Os \
    -fno-math-errno -ffreestanding -nostdlib -nostartfiles -Wl,-Ttext=0x10000 -Wl,--entry=0 \
    -o bench.elf taylor_kernels.c
echo "built bench.elf"
