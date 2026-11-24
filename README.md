# Overhead Assessment
Sigmoid, tanh and L2 normalization with MAED error detection.

* Sigmoid / tanh: 30 cached Maclaurin terms, summed once into even and odd parts
  (e^x = S_even + S_odd, e^-x = S_even - S_odd), so the check obtains its exponential
  for one extra addition. Inputs are clipped to [-3, 3].
* L2 normalization: check sum_i y_i^2 = 1.
* Tolerance 1e-4 (relative for sigmoid, absolute for tanh and L2).

## C (host)
```bash
gcc -o activation Overhead_Assessment/C/main.c -lm
./activation        # false-alarm sweep and fault-injection test
```

## Arduino UNO (ATmega328P @ 16 MHz)
`Overhead_Assessment/Arduino/main.ino` prints the time and clock cycles of 500 calls of each
baseline and protected function. Every result is written to a `volatile` sink; if the results
are discarded, `avr-gcc -Os` removes the computation and the check, and the measured overhead
is meaningless.

| Function | Clock cycles (500 calls) | Overhead |
|---|---|---|
| sigmoid | 12,087,168 | - |
| sigmoid with MAED | 12,711,872 | 5.2 % |
| tanh | 12,370,880 | - |
| tanh with MAED | 13,759,552 | 11.2 % |
| L2 norm (n = 128) | 28,040,576 | - |
| L2 norm with MAED | 45,981,632 | 64.0 % |

(arduino-cli, arduino:avr 1.8.6, cycle-accurate simulation of the ATmega328P.)

## Functions
- `sigmoid(x)`, `tanh_func(x)`, `l2norm(x, y, n)` - baseline
- `sigmoid_error_detection(x, &err)`, `tanh_error_detection(x, &err)`,
  `l2norm_error_detection(x, y, n, &err)` - with fault detection

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
