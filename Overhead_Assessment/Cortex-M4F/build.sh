#!/usr/bin/env bash
# Build bench.c for the Cortex-M4F (needs the GNU Arm Embedded toolchain, arm-none-eabi-gcc).
set -euo pipefail
cd "$(dirname "$0")"
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -Os \
    -fno-math-errno -ffreestanding -nostdlib -nostartfiles -Wl,-Ttext=0x10000 -Wl,--entry=0 \
    -o bench.elf bench.c
echo "built bench.elf"
