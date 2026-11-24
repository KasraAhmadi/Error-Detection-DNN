/*
 * MAED (Mathematical Activation Error Detection): sigmoid, tanh and L2
 * normalization with error detection. Same algorithm as the Arduino sketch
 * (Overhead_Assessment/Arduino/main.ino):
 *  - 30 cached Maclaurin terms, summed once into even and odd parts,
 *      e^{x} = S_even + S_odd,  e^{-x} = S_even - S_odd;
 *  - inputs clipped to [-3, 3];
 *  - tolerance 1e-4 (relative for sigmoid, absolute for tanh, sum y^2 = 1 for L2).
 * main() runs a false-alarm sweep and a fault-injection test.
 */
#include <stdio.h>
#include <math.h>

#define NUM_TERMS 30
#define EPSILON   0.0001

typedef enum { SUCCESS = 0, ERROR } ErrorCode;

static double maclaurin_array[NUM_TERMS];
static double S_even, S_odd;
static int fault = 0;                    /* test only: inject the DeepLaser output faults */

double clip_by_value(double x, double min_val, double max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

/* T_0 = 1, T_i = T_{i-1} * x / i */
void Maclaurin_terms(double x, int num_terms) {
    maclaurin_array[0] = 1;
    for (int i = 1; i < num_terms; i++)
        maclaurin_array[i] = maclaurin_array[i - 1] * x / i;
}

/* cached terms -> even and odd partial sums */
void maclaurin_sums(int num_terms) {
    double se = 0, so = 0;
    for (int i = 0; i < num_terms; i += 2) {
        se += maclaurin_array[i];
        so += maclaurin_array[i + 1];
    }
    S_even = se; S_odd = so;
}

double sigmoid(double input) {
    Maclaurin_terms(clip_by_value(input, -3, 3), NUM_TERMS);
    maclaurin_sums(NUM_TERMS);
    double inv_exp = S_even - S_odd;                 /* e^{-x} */
    return 1 / (1 + inv_exp);
}

double tanh_func(double input) {
    Maclaurin_terms(2 * clip_by_value(input, -3, 3), NUM_TERMS);
    maclaurin_sums(NUM_TERMS);
    double exp_value = S_even + S_odd;               /* e^{2x} */
    return (exp_value - 1) / (exp_value + 1);
}

/* h1(y) = y / (1 - y) = e^{x} */
double sigmoid_error_detection(double input, ErrorCode *error_out) {
    Maclaurin_terms(clip_by_value(input, -3, 3), NUM_TERMS);
    maclaurin_sums(NUM_TERMS);
    double inv_exp = S_even - S_odd;
    double y = 1 / (1 + inv_exp);
    if (fault) y = 1 - y;                            /* DeepLaser: negation skipped */
    /* error detection */
    double left = y / (1 - y);
    double right = S_even + S_odd;                   /* cached terms: one addition */
    *error_out = (fabs(left - right) < EPSILON * right) ? SUCCESS : ERROR;
    return y;
}

/* h2: alpha = (1 - y) / (1 + y) = e^{-2x}, compare (alpha-1)/(alpha+1) with (e^{-2x}-1)/(e^{-2x}+1) */
double tanh_error_detection(double input, ErrorCode *error_out) {
    Maclaurin_terms(2 * clip_by_value(input, -3, 3), NUM_TERMS);
    maclaurin_sums(NUM_TERMS);
    double exp_value = S_even + S_odd;
    double y = (exp_value - 1) / (exp_value + 1);
    if (fault) y = -y;                               /* DeepLaser: negation skipped */
    /* error detection */
    double exp_inv_value = S_even - S_odd;           /* cached terms: one addition */
    double alpha = (1 - y) / (1 + y);
    double left = (alpha - 1) / (alpha + 1);
    double right = (exp_inv_value - 1) / (exp_inv_value + 1);
    *error_out = (fabs(left - right) < EPSILON) ? SUCCESS : ERROR;
    return y;
}

/* y = x / ||x||_2 */
void l2norm(const double *x, double *y, int n) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    double nrm = sqrt(ss);
    if (nrm < 1e-12) nrm = 1e-12;
    double inv = 1 / nrm;
    for (int i = 0; i < n; i++) y[i] = x[i] * inv;
}

/* h4(y) = sum_i y_i^2 = 1 */
void l2norm_error_detection(const double *x, double *y, int n, ErrorCode *error_out) {
    l2norm(x, y, n);
    if (fault) for (int i = 0; i < n; i++) y[i] = 0; /* constant 1 of 1/||x|| lost -> y = 0 */
    double s = 0;
    for (int i = 0; i < n; i++) s += y[i] * y[i];
    *error_out = (fabs(s - 1) < EPSILON) ? SUCCESS : ERROR;
}

int main(void) {
    double x[128], y[128];
    for (fault = 0; fault <= 1; fault++) {
        int a_sig = 0, a_tanh = 0, a_l2 = 0;
        for (int i = 0; i <= 2000; i++) {
            ErrorCode e1, e2;
            double v = -10.0 + i * 0.01;
            sigmoid_error_detection(v, &e1);
            tanh_error_detection(v, &e2);
            a_sig += (e1 != SUCCESS); a_tanh += (e2 != SUCCESS);
        }
        for (int k = 0; k < 200; k++) {
            ErrorCode e3;
            for (int i = 0; i < 128; i++) x[i] = sin(0.37 * i + k) * (1 + k % 10);
            l2norm_error_detection(x, y, 128, &e3);
            a_l2 += (e3 != SUCCESS);
        }
        printf("%s: sigmoid %d/2001, tanh %d/2001, l2norm %d/200\n",
               fault ? "detected (faults injected)" : "false alarms (fault-free)", a_sig, a_tanh, a_l2);
    }
    return 0;
}
