/*
 * MAED overhead on the Arduino UNO (ATmega328P @ 16 MHz).
 *
 * Baseline vs. protected (MAED) sigmoid, tanh and L2 normalization.
 *  - Sigmoid / tanh: 30 cached Maclaurin terms, summed once into even and odd parts,
 *      e^{x} = S_even + S_odd,  e^{-x} = S_even - S_odd,
 *    so the check obtains its exponential for one extra addition.
 *    Inputs are clipped to [-3, 3]; on the AVR, double is 32-bit. The checks are
 *    absolute: |y/(1-y) - e^x| < 5e-3 (sigmoid) and 5e-4 (tanh), plus an input check
 *    (the clipped input is compared with an independent recomputation of clip(x)).
 *  - L2 normalization (n = 128): check sum_i y_i^2 = 1 with tolerance 1e-4.
 *  - Every result is written to a volatile sink: if results are discarded,
 *    avr-gcc -Os deletes the computation and the check, and the measured
 *    overhead is meaningless.
 * Prints the elapsed time / clock cycles of 500 calls per function, the number
 * of alarms during the timing runs, and a false-alarm sweep over [-10, 10].
 */
#include <Arduino.h>
#define SAMPLES 500



// Coefficients for Maclaurin series
double coeffs[] = {
   1.0, 
   1.0,
   0.5,
   0.16666666666666666,
   0.041666666666666664,
   0.008333333333333333,
   0.001388888888888889,
   0.0001984126984126984,
   2.48015873015873e-05,
   2.7557319223985893e-06,
   2.755731922398589e-07,
   2.505210838544172e-08,
   2.08767569878681e-09,
   1.6059043836821613e-10,
   1.1470745597729724e-11,
   7.647163731819816e-13,
   4.779477332387385e-14,
   2.8114572543455206e-15,
   1.5619206968586225e-16,
   8.22063524662433e-18,
   4.110317623312165e-19,
   1.9572941063391263e-20,
   8.896791392450574e-22,
   3.868170170630684e-23,
   1.6117375710961183e-24,
   6.446950284384473e-26,
   2.4795962632247973e-27,
   9.183689863795545e-29,
   3.279174951355552e-30,
   1.1307155004329145e-31,
};

typedef enum {
    SUCCESS = 0,
    ERROR
} ErrorCode;

// noinline + barrier: the check's recomputation of clip(x) is a separate, independent call
__attribute__((noinline)) double clip_by_value(double x, double min_val, double max_val) {
    __asm__ volatile("" ::: "memory");
    if (x < min_val)
        return min_val;
    else if (x > max_val)
        return max_val;
    else
        return x;
}

double maclaurin_array[30];

void Maclaurin_terms(double x, int num_terms) {
    maclaurin_array[0] = 1;
    maclaurin_array[1] = x / coeffs[0];
    for (int i = 2; i < num_terms; i++){
        maclaurin_array[i] = maclaurin_array[i-1] * x / (i); 
    }
}

// Cached Maclaurin terms are summed once into even and odd parts:
//   e^{x} = S_even + S_odd,   e^{-x} = S_even - S_odd
// so the baseline gets the exponential it needs and the check gets the
// other one for a single extra addition (as in the FPGA design).
double S_even, S_odd;
void maclaurin_sums(int num_terms) {
    double se = 0, so = 0;
    for (int i = 0; i < num_terms; i += 2) {
        se += maclaurin_array[i];
        so += maclaurin_array[i + 1];
    }
    S_even = se; S_odd = so;
}

double sigmoid(double input){
    double clipped_value = clip_by_value(input, -3, 3);
    int arraySize = 30;
    Maclaurin_terms(clipped_value, arraySize);
    maclaurin_sums(arraySize);
    double inv_exp = S_even - S_odd;
    return 1 / (1 + inv_exp);
}

double tanh_func(double input){
    double clipped_value = clip_by_value(input, -3, 3);
    int arraySize = 30;
    Maclaurin_terms(2 * clipped_value, arraySize);
    maclaurin_sums(arraySize);
    double exp_value = S_even + S_odd;
    return (exp_value - 1) / (exp_value + 1);
}

double tanh_error_detection(double input, ErrorCode *error_out){
    double clipped_value = clip_by_value(input, -3, 3);
    int arraySize = 30;
    Maclaurin_terms(2 * clipped_value, arraySize);
    maclaurin_sums(arraySize);
    double exp_value = S_even + S_odd;
    double y = (exp_value - 1) / (exp_value + 1);

    // Error detection started
    double exp_inv_value = S_even - S_odd;        // cached terms: one addition
    double alpha = (1 - y) / (1 + y);
    double left = (alpha - 1) / (alpha + 1);
    double right = (exp_inv_value - 1) / (exp_inv_value + 1);
    int input_bad = (clipped_value != clip_by_value(input, -3, 3));   // input check
    double epsilon = 0.0005;            // float32: absolute tolerance (worst fault-free residual 1.5e-4)
    if (fabs(left - right) < epsilon && !input_bad) {
        *error_out = SUCCESS;
        return y;
    } else {
        *error_out = ERROR;
        return 0;
    }
}

double sigmoid_error_detection(double input, ErrorCode *error_out){
    double clipped_value = clip_by_value(input, -3, 3);
    int arraySize = 30;
    Maclaurin_terms(clipped_value, arraySize);
    maclaurin_sums(arraySize);
    double inv_exp = S_even - S_odd;
    double y = 1 / (1 + inv_exp);

    // Error detection started
    double left = y / (1 - y);
    double right = S_even + S_odd;                // cached terms: one addition
    int input_bad = (clipped_value != clip_by_value(input, -3, 3));   // input check
    double epsilon = 0.005;             // float32: absolute tolerance (worst fault-free residual 1.2e-3)
    if (fabs(left - right) < epsilon && !input_bad) {
        *error_out = SUCCESS;
        return y;
    } else {
        *error_out = ERROR;
        return 0;
    }
}

const double test_inputs[SAMPLES] PROGMEM = {
  3.45, -7.82, 1.23, -9.54, 5.67, -2.34, 8.91, -4.56, 0.78, -6.89,
  9.12, -3.45, 2.56, -8.76, 4.32, -1.23, 7.89, -5.67, 3.21, -9.87,
  6.54, -4.32, 1.09, -7.65, 8.34, -2.10, 5.43, -6.78, 9.00, -3.56,
  4.67, -8.23, 2.34, -5.89, 7.56, -1.45, 6.78, -9.34, 3.90, -4.67,
  8.12, -2.56, 1.34, -7.23, 5.89, -3.78, 9.45, -6.12, 4.56, -8.90,
  2.78, -5.34, 7.01, -1.56, 6.23, -9.67, 3.67, -4.12, 8.56, -2.89,
  1.78, -7.90, 5.12, -3.23, 9.34, -6.45, 4.23, -8.67, 2.90, -5.78,
  7.34, -1.34, 6.56, -9.23, 3.12, -4.89, 8.67, -2.34, 1.45, -7.01,
  5.78, -3.89, 9.56, -6.34, 4.01, -8.12, 2.67, -5.67, 7.45, -1.23,
  6.12, -9.34, 3.89, -4.45, 8.34, -2.78, 1.23, -7.45, 5.45, -3.12,
 3.45, -7.82, 1.23, -9.54, 5.67, -2.34, 8.91, -4.56, 0.78, -6.89,
  9.12, -3.45, 2.56, -8.76, 4.32, -1.23, 7.89, -5.67, 3.21, -9.87,
  6.54, -4.32, 1.09, -7.65, 8.34, -2.10, 5.43, -6.78, 9.00, -3.56,
  4.67, -8.23, 2.34, -5.89, 7.56, -1.45, 6.78, -9.34, 3.90, -4.67,
  8.12, -2.56, 1.34, -7.23, 5.89, -3.78, 9.45, -6.12, 4.56, -8.90,
  2.78, -5.34, 7.01, -1.56, 6.23, -9.67, 3.67, -4.12, 8.56, -2.89,
  1.78, -7.90, 5.12, -3.23, 9.34, -6.45, 4.23, -8.67, 2.90, -5.78,
  7.34, -1.34, 6.56, -9.23, 3.12, -4.89, 8.67, -2.34, 1.45, -7.01,
  5.78, -3.89, 9.56, -6.34, 4.01, -8.12, 2.67, -5.67, 7.45, -1.23,
  6.12, -9.34, 3.89, -4.45, 8.34, -2.78, 1.23, -7.45, 5.45, -3.12,
    3.45, -7.82, 1.23, -9.54, 5.67, -2.34, 8.91, -4.56, 0.78, -6.89,
  9.12, -3.45, 2.56, -8.76, 4.32, -1.23, 7.89, -5.67, 3.21, -9.87,
  6.54, -4.32, 1.09, -7.65, 8.34, -2.10, 5.43, -6.78, 9.00, -3.56,
  4.67, -8.23, 2.34, -5.89, 7.56, -1.45, 6.78, -9.34, 3.90, -4.67,
  8.12, -2.56, 1.34, -7.23, 5.89, -3.78, 9.45, -6.12, 4.56, -8.90,
  2.78, -5.34, 7.01, -1.56, 6.23, -9.67, 3.67, -4.12, 8.56, -2.89,
  1.78, -7.90, 5.12, -3.23, 9.34, -6.45, 4.23, -8.67, 2.90, -5.78,
  7.34, -1.34, 6.56, -9.23, 3.12, -4.89, 8.67, -2.34, 1.45, -7.01,
  5.78, -3.89, 9.56, -6.34, 4.01, -8.12, 2.67, -5.67, 7.45, -1.23,
  6.12, -9.34, 3.89, -4.45, 8.34, -2.78, 1.23, -7.45, 5.45, -3.12,
      0.45, -7.82, 1.23, -9.54, 5.67, -2.34, 8.91, -4.56, 0.78, -6.89,
  0.12, -0.45, 2.56, -8.76, 4.32, -1.23, 7.89, -5.67, 3.21, -9.87,
  6.54, -0.32, 1.09, -7.65, 8.34, -2.10, 5.43, -6.78, 9.00, -3.56,
  4.67, -0.23, 2.34, -5.89, 7.56, -1.45, 6.78, -9.34, 3.90, -4.67,
  8.12, -0.56, 1.34, -7.23, 5.89, -3.78, 9.45, -6.12, 4.56, -8.90,
  0.78, -0.34, 7.01, -1.56, 6.23, -9.67, 3.67, -4.12, 8.56, -2.89,
  1.78, -0.90, 5.12, -3.23, 9.34, -6.45, 4.23, -8.67, 2.90, -5.78,
  7.34, -0.34, 6.56, -9.23, 3.12, -4.89, 8.67, -2.34, 1.45, -7.01,
  5.78, -0.89, 9.56, -6.34, 4.01, -8.12, 2.67, -5.67, 7.45, -1.23,
  6.12, -0.34, 3.89, -4.45, 8.34, -2.78, 1.23, -7.45, 5.45, -3.12,
      3.45, -7.82, 1.23, -9.54, 5.67, -2.34, 8.91, -4.56, 0.78, -6.89,
  0.12, -0.45, 2.56, -8.76, 4.32, -1.23, 7.89, -5.67, 3.21, -9.87,
  0.54, -0.32, 1.09, -7.65, 8.34, -2.10, 5.43, -6.78, 9.00, -3.56,
  0.67, -0.23, 2.34, -5.89, 7.56, -1.45, 6.78, -9.34, 3.90, -4.67,
  0.12, -0.56, 1.34, -7.23, 5.89, -3.78, 9.45, -6.12, 4.56, -8.90,
  0.78, -0.34, 7.01, -1.56, 6.23, -9.67, 3.67, -4.12, 8.56, -2.89,
  0.78, -0.90, 5.12, -3.23, 9.34, -6.45, 4.23, -8.67, 2.90, -5.78,
  0.34, -0.34, 6.56, -9.23, 3.12, -4.89, 8.67, -2.34, 1.45, -7.01,
  0.78, -0.89, 9.56, -6.34, 4.01, -8.12, 2.67, -5.67, 7.45, -1.23,
  0.12, -0.34, 3.89, -4.45, 8.34, -2.78, 1.23, -7.45, 5.45, -3.12,
};

// Read test input from PROGMEM
double getTestInput(int index) {
    return pgm_read_float(&test_inputs[index]);
}
int input_index = 0;

// ---------------------------------------------------------------------------
// Sinks: make every result observable, so that avr-gcc -Os cannot delete the
// computation (or the error check) whose result would otherwise be unused.
volatile double sink;
volatile unsigned int err_count;

// ---------------------------------------------------------------------------
// L2 normalization y = x / ||x||_2  and its MAED check  sum_i y_i^2 = 1
#define L2_DIM 128
double l2_x[L2_DIM + 8];
double l2_y[L2_DIM];

void l2norm(const double *x, double *y, int n) {
    double ss = 0;
    for (int i = 0; i < n; i++)
        ss += x[i] * x[i];
    double nrm = sqrt(ss);
    if (nrm < 1e-12)
        nrm = 1e-12;
    double inv = 1 / nrm;
    for (int i = 0; i < n; i++)
        y[i] = x[i] * inv;
}

void l2norm_error_detection(const double *x, double *y, int n, ErrorCode *error_out) {
    l2norm(x, y, n);
    // Error detection started
    double s = 0;
    for (int i = 0; i < n; i++)
        s += y[i] * y[i];
    double epsilon = 0.0001;          // float32 (AVR double): sum of n rounded squares
    if (fabs(s - 1) < epsilon)
        *error_out = SUCCESS;
    else
        *error_out = ERROR;
}

unsigned long run_time(int which) {
    unsigned long t = micros();
    for (input_index = 0; input_index < SAMPLES; input_index++) {
        ErrorCode err = SUCCESS;
        double x = getTestInput(input_index);
        switch (which) {
            case 0: sink = sigmoid(x); break;
            case 1: sink = sigmoid_error_detection(x, &err); break;
            case 2: sink = tanh_func(x); break;
            case 3: sink = tanh_error_detection(x, &err); break;
            case 4: l2norm(l2_x + (input_index & 7), l2_y, L2_DIM); sink = l2_y[input_index & 63]; break;
            case 5: l2norm_error_detection(l2_x + (input_index & 7), l2_y, L2_DIM, &err); sink = l2_y[input_index & 63]; break;
        }
        if (err != SUCCESS) err_count++;
    }
    return micros() - t;
}

const char *names[] = {"sigmoid", "error_sigmoid", "tanh", "error_tanh", "l2norm", "error_l2norm"};

void setup() {
    Serial.begin(9600);
    for (int i = 0; i < L2_DIM + 8; i++)          // L2 input vector from the same test inputs
        l2_x[i] = getTestInput((i * 37) % SAMPLES);
    Serial.println("Arduino UNO clock cycles (results used, volatile sink)");
    // false-alarm sweep: 2001 inputs in [-10, 10] for sigmoid / tanh, 200 random-ish vectors for L2
    unsigned int fa_sig = 0, fa_tanh = 0, fa_l2 = 0;
    for (int i = 0; i <= 2000; i++) {
        double x = -10.0 + i * 0.01;
        ErrorCode e1, e2;
        sigmoid_error_detection(x, &e1); tanh_error_detection(x, &e2);
        fa_sig += (e1 != SUCCESS); fa_tanh += (e2 != SUCCESS);
    }
    for (int k = 0; k < 200; k++) {
        ErrorCode e3;
        for (int i = 0; i < L2_DIM; i++) l2_y[i] = getTestInput((i * 7 + k * 13) % SAMPLES) * (1 + (k % 10));
        for (int i = 0; i < L2_DIM; i++) l2_x[i] = l2_y[i];
        l2norm_error_detection(l2_x, l2_y, L2_DIM, &e3);
        fa_l2 += (e3 != SUCCESS);
    }
    for (int i = 0; i < L2_DIM + 8; i++) l2_x[i] = getTestInput((i * 37) % SAMPLES);
    Serial.print("false alarms (sweep): sigmoid "); Serial.print(fa_sig); Serial.print("/2001, tanh ");
    Serial.print(fa_tanh); Serial.print("/2001, l2norm "); Serial.print(fa_l2); Serial.println("/200");
}

void loop() {
    for (int w = 0; w < 6; w++) {
        err_count = 0;
        unsigned long us = run_time(w);
        Serial.print(names[w]); Serial.print(": ");
        Serial.print(us); Serial.print(" us, ");
        Serial.print(us * (F_CPU / 1000000UL)); Serial.print(" cycles, alarms ");
        Serial.println(err_count);
    }
    Serial.println("========== END ==========");
    while (1) {}
}
