/*
 * Algorithm-5 (paper) implementation of the exponent-based functions, used for EVERYTHING:
 * the MobileCLIP2 attacks (Sec. III), the MAED detection (Sec. IV), and the Cortex-M4F
 * overhead benchmark (Sec. VI).  Same arithmetic as the Arduino sketch.
 *
 *   x_c = clip(x, -3, 3);  T_0 = 1, T_i = T_{i-1} * x_c / i  (N = 30 cached terms)
 *   S_e = sum of even terms, S_o = sum of odd terms
 *   sigmoid:  y = 1 / (1 + (S_e - S_o))          check  |y/(1-y) - (S_e + S_o)| < 5e-3
 *   tanh(u):  terms of 2*u_c, E = S_e + S_o, t = (E-1)/(E+1)
 *             check |(a-1)/(a+1) - (I-1)/(I+1)| < 5e-4,  a = (1-t)/(1+t),  I = S_e - S_o
 *   GELU:     y = 0.5 v (1 + tanh(u)),  u = 0.797885 (v + 0.044715 v^3);  the check recovers
 *             t = 2y/v - 1 from the output, so it also covers the outer factor.
 *   input check: the core functions also return the clipped input x_c; the check recomputes
 *             clip(x) independently and flags x_c != clip(x) (a fault in the clamp runs before the
 *             cached terms are generated and would otherwise be a common-mode fault).
 * float = 32-bit (as the AVR double).  The protected kernels call the same core functions as
 * the unprotected ones, so the attacked instructions are identical.
 * Build: see build_taylor.sh (-Os, Cortex-M4F, FPv4-SP).
 */
#include <stdint.h>
#define KERNEL __attribute__((noinline, used))
#define NT 30
#define EPS_SIG 0.005f
#define EPS_TANH 0.0005f
#define EPS_L2 0.0001f

static inline float clip3(float x)
{ if (x < -3.0f) return -3.0f; else if (x > 3.0f) return 3.0f; else return x; }

/* cached Maclaurin terms of e^z and their even / odd partial sums */
KERNEL void maclaurin_sums(float z, float *se_o, float *so_o)
{
    float t[NT];
    t[0] = 1;
    t[1] = z / 1.0f;
    for (int i = 2; i < NT; i++)
        t[i] = t[i - 1] * z / (i);
    float se = 0, so = 0;
    for (int i = 0; i < NT; i += 2) { se += t[i]; so += t[i + 1]; }
    *se_o = se; *so_o = so;
}

/* ---------------------------------------------------------------- core functions (attacked) */
KERNEL float sig_core(float x, float *se, float *so, float *xc)
{
    float c = clip3(x);
    *xc = c;
    maclaurin_sums(c, se, so);
    float inv_exp = *se - *so;                         /* e^{-x} */
    return 1 / (1 + inv_exp);
}

KERNEL float tanh_core(float u, float *se, float *so, float *uc)
{
    float c = clip3(u);
    *uc = c;
    maclaurin_sums(2 * c, se, so);
    float e = *se + *so;                               /* e^{2u} */
    return (e - 1) / (e + 1);
}

/* argument of the tanh in GELU; shared by gelu_core and the check (bit-identical) */
KERNEL float gelu_u(float v)
{ return 0.797885f * (v + 0.044715f * v * v * v); }

KERNEL float gelu_core(float v, float *se, float *so, float *uc)
{
    float u = gelu_u(v);
    float t = tanh_core(u, se, so, uc);
    return 0.5f * v * (1 + t);
}

/* ---------------------------------------------------------------- checks (MAED) */
static inline int sig_bad(float y, float se, float so)
{
    float left = y / (1 - y);
    float right = se + so;                             /* e^{x}: one addition */
    return !(__builtin_fabsf(left - right) < EPS_SIG);  /* NaN -> detected */
}

static inline int tanh_bad(float t, float se, float so)
{
    float alpha = (1 - t) / (1 + t);
    float left = (alpha - 1) / (alpha + 1);
    float i = se - so;                                 /* e^{-2u}: one addition */
    float right = (i - 1) / (i + 1);
    return !(__builtin_fabsf(left - right) < EPS_TANH);
}

/* ---------------------------------------------------------------- MobileCLIP2 kernels */
KERNEL void sigmoid_t_f32(const float *x, float *y, int n)
{ float se, so, xc; for (int i = 0; i < n; i++) y[i] = sig_core(x[i], &se, &so, &xc); }

KERNEL int sigmoid_t_maed_f32(const float *x, float *y, int n)
{
    int bad = 0; float se, so, xc;
    for (int i = 0; i < n; i++) {
        float v = sig_core(x[i], &se, &so, &xc);
        y[i] = v;
        bad += sig_bad(v, se, so) | (xc != clip3(x[i]));     /* identity + input check */
    }
    return bad;
}

KERNEL void gelu_t_f32(const float *x, float *y, int n)
{ float se, so, uc; for (int i = 0; i < n; i++) y[i] = gelu_core(x[i], &se, &so, &uc); }

KERNEL int gelu_t_maed_f32(const float *x, float *y, int n)
{
    int bad = 0; float se, so, uc;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float out = gelu_core(v, &se, &so, &uc);
        y[i] = out;
        int in_bad = (uc != clip3(gelu_u(v)));          /* input check */
        if (v == 0.0f) { bad += in_bad | !(out == 0.0f); continue; }
        float t = 2.0f * out / v - 1.0f;               /* tanh recovered from the GELU output */
        bad += tanh_bad(t, se, so) | in_bad;
    }
    return bad;
}

/* ---------------------------------------------------------------- overhead benchmark
 * Same structure as the Arduino sketch: 500 calls, results to a sink, return = #alarms. */
KERNEL int bench_sigmoid(const float *in, float *out, int n)
{ float se, so, xc; for (int i = 0; i < n; i++) out[i] = sig_core(in[i], &se, &so, &xc); return 0; }

KERNEL int bench_error_sigmoid(const float *in, float *out, int n)
{
    int a = 0; float se, so, xc;
    for (int i = 0; i < n; i++) {
        float y = sig_core(in[i], &se, &so, &xc);
        out[i] = y;
        a += sig_bad(y, se, so) | (xc != clip3(in[i]));
    }
    return a;
}

KERNEL int bench_tanh(const float *in, float *out, int n)
{ float se, so, uc; for (int i = 0; i < n; i++) out[i] = tanh_core(in[i], &se, &so, &uc); return 0; }

KERNEL int bench_error_tanh(const float *in, float *out, int n)
{
    int a = 0; float se, so, uc;
    for (int i = 0; i < n; i++) {
        float t = tanh_core(in[i], &se, &so, &uc);
        out[i] = t;
        a += tanh_bad(t, se, so) | (uc != clip3(in[i]));
    }
    return a;
}

#define L2_DIM 128
KERNEL void l2norm_core(const float *x, float *y, int n)
{
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float nrm = __builtin_sqrtf(ss);
    if (nrm < 1e-12f) nrm = 1e-12f;
    float inv = 1 / nrm;
    for (int i = 0; i < n; i++) y[i] = x[i] * inv;
}

KERNEL int bench_l2norm(const float *xbuf, float *ybuf, float *sink, int n)
{ for (int i = 0; i < n; i++) { l2norm_core(xbuf + (i & 7), ybuf, L2_DIM); sink[i] = ybuf[i & 63]; } return 0; }

KERNEL int bench_error_l2norm(const float *xbuf, float *ybuf, float *sink, int n)
{
    int a = 0;
    for (int i = 0; i < n; i++) {
        l2norm_core(xbuf + (i & 7), ybuf, L2_DIM);
        float s = 0;
        for (int j = 0; j < L2_DIM; j++) s += ybuf[j] * ybuf[j];
        a += !(__builtin_fabsf(s - 1) < EPS_L2);
        sink[i] = ybuf[i & 63];
    }
    return a;
}

/* ---------------------------------------------------------------- L2 normalization (MobileCLIP2 head)
 * same l2norm_core as the overhead benchmark; eps argument kept for the fault_eval head interface */
KERNEL void l2norm_t_f32(const float *x, float *y, int n, float eps)
{ (void)eps; l2norm_core(x, y, n); }

KERNEL int l2norm_t_maed_f32(const float *x, float *y, int n, float eps)
{
    (void)eps;
    l2norm_core(x, y, n);
    float s = 0;
    for (int i = 0; i < n; i++) s += y[i] * y[i];
    return !(__builtin_fabsf(s - 1) < EPS_L2);
}
