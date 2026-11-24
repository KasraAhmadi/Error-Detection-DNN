/* MAED overhead benchmark for Cortex-M4F: the same code as the Arduino UNO sketch
 * (Overhead_Assessment/Arduino/main.ino): 30 cached Maclaurin terms summed into even/odd parts,
 * inputs clipped to [-3, 3], absolute checks (sigmoid eps 1e-3, tanh 1e-4, L2 1e-4),
 * L2 vector length 128, 500 calls on the same 500 test inputs.
 * float = 32-bit, as the AVR double. Results go to an output buffer (no dead code).
 * The term buffer is on the stack (the emulator maps no .data/.bss); arithmetic unchanged. */
#include <stdint.h>
#define KERNEL __attribute__((noinline, used))
#define NT 30
#define SAMPLES 500
typedef enum { SUCCESS = 0, ERROR } ErrorCode;

static inline float clip_by_value(float x, float lo, float hi)
{ if (x < lo) return lo; else if (x > hi) return hi; else return x; }

__attribute__((noinline)) static void Maclaurin_terms(float x, int n, float *t)
{
    t[0] = 1;
    t[1] = x / 1.0f;
    for (int i = 2; i < n; i++)
        t[i] = t[i - 1] * x / (i);
}

__attribute__((noinline)) static void maclaurin_sums(const float *t, int n, float *se_o, float *so_o)
{
    float se = 0, so = 0;
    for (int i = 0; i < n; i += 2) { se += t[i]; so += t[i + 1]; }
    *se_o = se; *so_o = so;
}

__attribute__((noinline)) static float sigmoid(float input)
{
    float t[NT], se, so;
    Maclaurin_terms(clip_by_value(input, -3, 3), NT, t);
    maclaurin_sums(t, NT, &se, &so);
    float inv_exp = se - so;
    return 1 / (1 + inv_exp);
}

__attribute__((noinline)) static float tanh_func(float input)
{
    float t[NT], se, so;
    Maclaurin_terms(2 * clip_by_value(input, -3, 3), NT, t);
    maclaurin_sums(t, NT, &se, &so);
    float exp_value = se + so;
    return (exp_value - 1) / (exp_value + 1);
}

__attribute__((noinline)) static float sigmoid_error_detection(float input, ErrorCode *err)
{
    float t[NT], se, so;
    Maclaurin_terms(clip_by_value(input, -3, 3), NT, t);
    maclaurin_sums(t, NT, &se, &so);
    float inv_exp = se - so;
    float y = 1 / (1 + inv_exp);
    float left = y / (1 - y);
    float right = se + so;                       /* cached terms: one addition */
    float epsilon = 0.001f;                      /* absolute */
    if (__builtin_fabsf(left - right) < epsilon) { *err = SUCCESS; return y; }
    *err = ERROR; return 0;
}

__attribute__((noinline)) static float tanh_error_detection(float input, ErrorCode *err)
{
    float t[NT], se, so;
    Maclaurin_terms(2 * clip_by_value(input, -3, 3), NT, t);
    maclaurin_sums(t, NT, &se, &so);
    float exp_value = se + so;
    float y = (exp_value - 1) / (exp_value + 1);
    float exp_inv_value = se - so;               /* cached terms: one addition */
    float alpha = (1 - y) / (1 + y);
    float left = (alpha - 1) / (alpha + 1);
    float right = (exp_inv_value - 1) / (exp_inv_value + 1);
    float epsilon = 0.0001f;                     /* absolute */
    if (__builtin_fabsf(left - right) < epsilon) { *err = SUCCESS; return y; }
    *err = ERROR; return 0;
}

#define L2_DIM 128
__attribute__((noinline)) static void l2norm(const float *x, float *y, int n)
{
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float nrm = __builtin_sqrtf(ss);
    if (nrm < 1e-12f) nrm = 1e-12f;
    float inv = 1 / nrm;
    for (int i = 0; i < n; i++) y[i] = x[i] * inv;
}

__attribute__((noinline)) static void l2norm_error_detection(const float *x, float *y, int n, ErrorCode *err)
{
    l2norm(x, y, n);
    float s = 0;
    for (int i = 0; i < n; i++) s += y[i] * y[i];
    *err = (__builtin_fabsf(s - 1) < 0.0001f) ? SUCCESS : ERROR;
}

/* ---- benchmarks: 500 calls each; out[] is the sink; return = number of alarms ---- */
KERNEL int bench_sigmoid(const float *in, float *out, int n)
{ for (int i = 0; i < n; i++) out[i] = sigmoid(in[i]); return 0; }

KERNEL int bench_error_sigmoid(const float *in, float *out, int n)
{ int a = 0; for (int i = 0; i < n; i++) { ErrorCode e = SUCCESS; out[i] = sigmoid_error_detection(in[i], &e); a += (e != SUCCESS); } return a; }

KERNEL int bench_tanh(const float *in, float *out, int n)
{ for (int i = 0; i < n; i++) out[i] = tanh_func(in[i]); return 0; }

KERNEL int bench_error_tanh(const float *in, float *out, int n)
{ int a = 0; for (int i = 0; i < n; i++) { ErrorCode e = SUCCESS; out[i] = tanh_error_detection(in[i], &e); a += (e != SUCCESS); } return a; }

/* xbuf: L2_DIM + 8 values; ybuf: L2_DIM values; sink: n values */
KERNEL int bench_l2norm(const float *xbuf, float *ybuf, float *sink, int n)
{ for (int i = 0; i < n; i++) { l2norm(xbuf + (i & 7), ybuf, L2_DIM); sink[i] = ybuf[i & 63]; } return 0; }

KERNEL int bench_error_l2norm(const float *xbuf, float *ybuf, float *sink, int n)
{ int a = 0; for (int i = 0; i < n; i++) { ErrorCode e = SUCCESS; l2norm_error_detection(xbuf + (i & 7), ybuf, L2_DIM, &e); sink[i] = ybuf[i & 63]; a += (e != SUCCESS); } return a; }
