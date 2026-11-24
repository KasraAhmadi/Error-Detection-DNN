# Overhead Assessment
Sigmoid, tanh and L2 normalization with MAED error detection, and their clock-cycle overhead
on an **ATmega328P** (Arduino UNO) and an **ARM Cortex-M4F**. Both use the same code and the
same assumptions:

* 32-bit floating point (the AVR `double` is 32-bit; the Cortex-M4F uses `float`).
* Sigmoid / tanh: 30 Maclaurin terms, cached and summed once into even and odd parts
  (`e^x = S_even + S_odd`, `e^-x = S_even - S_odd`), so the check obtains its exponential for
  one extra addition. Inputs are clipped to [-3, 3].
* L2 normalization (vector length 128): check `sum_i y_i^2 = 1`.
* Absolute checks: `|y/(1-y) - e^x| < 1e-3` (sigmoid), `1e-4` (tanh), `|sum y^2 - 1| < 1e-4` (L2).
* 500 calls per function on the same 500 test inputs. Every result is written to a sink
  (`volatile` on the AVR). If the results are discarded, the compiler removes the computation
  and the check, and the measured overhead is meaningless.

| Function | ATmega328P cycles | Overhead | Cortex-M4F cycles | Overhead |
|---|---|---|---|---|
| sigmoid | 12,087,296 | - | 485,566 | - |
| sigmoid with MAED | 12,663,552 | 4.8 % | 506,574 | 4.3 % |
| tanh | 12,370,816 | - | 486,566 | - |
| tanh with MAED | 13,759,552 | 11.2 % | 524,074 | 7.7 % |
| L2 norm (n = 128) | 28,040,704 | - | 1,180,524 | - |
| L2 norm with MAED | 45,981,504 | 64.0 % | 1,775,033 | 50.4 % |

No false alarm occurs on either platform.

## 1. C (host): correctness check
```bash
gcc -o activation Overhead_Assessment/C/main.c -lm
./activation        # false-alarm sweep and fault-injection test
```
Expected output: 0 false alarms; injected faults detected (2000/2001, 2000/2001, 200/200;
only x = 0 is unaffected by the sigmoid/tanh faults).

## 2. ATmega328P (Arduino UNO)
`Overhead_Assessment/Arduino/main.ino` prints a false-alarm sweep and, for each function, the
time (`micros()`) and clock cycles of 500 calls on the serial port (9600 baud).

**On the board:** open `main.ino` in the Arduino IDE (it asks to move the file into a folder
named `main`; accept), select *Arduino Uno*, upload, and open the Serial Monitor.

**Without a board** (cycle-accurate simulation of the ATmega328P at 16 MHz with
[avr8js](https://github.com/wokwi/avr8js); Timer0 is simulated, so `micros()` behaves as on the board):
```bash
# requires arduino-cli with the arduino:avr core (1.8.6) and Node.js >= 18
arduino-cli core install arduino:avr
# arduino-cli needs the sketch file to have the same name as its folder
mkdir -p Overhead_Assessment/Arduino/build/main
cp Overhead_Assessment/Arduino/main.ino Overhead_Assessment/Arduino/build/main/
arduino-cli compile --fqbn arduino:avr:uno --output-dir Overhead_Assessment/Arduino/build Overhead_Assessment/Arduino/build/main
cd Overhead_Assessment/Arduino/simulate
npm install --ignore-scripts
node run.mjs ../build/main.ino.hex "== END =="
```

## 3. ARM Cortex-M4F
`Overhead_Assessment/Cortex-M4F/bench.c` contains the same functions (and reads the same 500
test inputs from `main.ino`). It is compiled with `-Os` for the Cortex-M4F with the
single-precision FPU and executed instruction by instruction in the Unicorn emulator
(`arm_emu.py`). The clock cycles are estimated from the executed instructions with the
instruction timings of the Cortex-M4 Technical Reference Manual, assuming zero-wait-state memory.
```bash
# requires the GNU Arm Embedded toolchain (arm-none-eabi-gcc) and Python 3
pip install unicorn capstone pyelftools numpy
cd Overhead_Assessment/Cortex-M4F
./build.sh
python run_bench.py
```

## Functions
- `sigmoid(x)`, `tanh_func(x)`, `l2norm(x, y, n)`: baseline
- `sigmoid_error_detection(x, &err)`, `tanh_error_detection(x, &err)`,
  `l2norm_error_detection(x, y, n, &err)`: with fault detection

# Training
# NASA Jet Engine RUL Prediction Enabled with Error Detection

CNN model to predict Remaining Useful Life of aircraft jet engines using NASA C-MAPSS dataset.

## Quick Start

### 1. Setup Environment
```bash
cd Training
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Data
Download NASA C-MAPSS dataset and place in `archive/CMaps/`:
- `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`
- `train_FD002.txt`, `test_FD002.txt`, `RUL_FD002.txt`
- `train_FD003.txt`, `test_FD003.txt`, `RUL_FD003.txt`
- `train_FD004.txt`, `test_FD004.txt`, `RUL_FD004.txt`

### 3. Run
```bash
python training.py
```

## Project Structure
```
nasa-rul-prediction/
├── README.md
├── requirements.txt
├── training.py
├── models/              # Output
└── archive/CMaps/       # Dataset folder
```

## Requirements
- Python 3.12.3
- See `requirements.txt`


## Results
Models saved to `models/` folder with metrics (Accuracy, F1-score, Recall) and average inference time.
