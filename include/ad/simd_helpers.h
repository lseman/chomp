// simd_helpers.h — AVX2-accelerated lane loops for autodiff
// Process 4 lanes (8 for AVX-512) at once in forward_dot_lanes / backward_lanes
#pragma once

#if defined(__AVX2__) && !defined(AD_DISABLE_SIMD)
#define AD_HAS_SIMD 1
#include <immintrin.h>

namespace ad::simd {

// =====================================================================
// AVX2 helpers (4 doubles @ 256-bit)
// =====================================================================

// Vectorized fill: dot[ybase .. ybase+3] = val
inline void vfload4(double* __restrict dst, double val) noexcept {
    __m256d v = _mm256_set1_pd(val);
    _mm256_storeu_pd(dst, v);
}

// Vectorized fill: dot[ybase .. ybase+n-1] = 0
inline void vfill_n(double* __restrict dst, size_t n) noexcept {
    __m256d z = _mm256_setzero_pd();
    for (size_t i = 0; i + 4 <= n; i += 4)
        _mm256_storeu_pd(&dst[i], z);
    for (size_t i = (n & ~3); i < n; ++i)
        dst[i] = 0.0;
}

// Vectorized fill: dot[ybase .. ybase+n-1] = val
inline void vfill_n_val(double* __restrict dst, size_t n, double val) noexcept {
    __m256d v = _mm256_set1_pd(val);
    for (size_t i = 0; i + 4 <= n; i += 4)
        _mm256_storeu_pd(&dst[i], v);
    for (size_t i = (n & ~3); i < n; ++i)
        dst[i] = val;
}

// Unary: dot[ybase+l] = coeff * dot[xbase+l]   for l in [0, n)
inline void vscal4(double* __restrict dst, const double* __restrict src,
                   size_t n, double coeff) noexcept {
    __m256d c = _mm256_set1_pd(coeff);
    for (size_t i = 0; i + 4 <= n; i += 4)
        _mm256_storeu_pd(&dst[i], _mm256_mul_pd(_mm256_loadu_pd(&src[i]), c));
    for (size_t i = (n & ~3); i < n; ++i)
        dst[i] = coeff * src[i];
}

// Binary: dst[l] = srcA[l] op srcB[l]   for l in [0, n)
inline void vadd4(double* __restrict dst, const double* __restrict a,
                  const double* __restrict b, size_t n) noexcept {
    for (size_t i = 0; i + 4 <= n; i += 4)
        _mm256_storeu_pd(&dst[i],
            _mm256_add_pd(_mm256_loadu_pd(&a[i]), _mm256_loadu_pd(&b[i])));
    for (size_t i = (n & ~3); i < n; ++i)
        dst[i] = a[i] + b[i];
}

inline void vsub4(double* __restrict dst, const double* __restrict a,
                  const double* __restrict b, size_t n) noexcept {
    for (size_t i = 0; i + 4 <= n; i += 4)
        _mm256_storeu_pd(&dst[i],
            _mm256_sub_pd(_mm256_loadu_pd(&a[i]), _mm256_loadu_pd(&b[i])));
    for (size_t i = (n & ~3); i < n; ++i)
        dst[i] = a[i] - b[i];
}

inline void vmul4(double* __restrict dst, const double* __restrict a,
                  const double* __restrict b, size_t n) noexcept {
    for (size_t i = 0; i + 4 <= n; i += 4)
        _mm256_storeu_pd(&dst[i],
            _mm256_mul_pd(_mm256_loadu_pd(&a[i]), _mm256_loadu_pd(&b[i])));
    for (size_t i = (n & ~3); i < n; ++i)
        dst[i] = a[i] * b[i];
}

inline void vdiv4(double* __restrict dst, const double* __restrict a,
                  const double* __restrict b, size_t n) noexcept {
    for (size_t i = 0; i + 4 <= n; i += 4)
        _mm256_storeu_pd(&dst[i],
            _mm256_div_pd(_mm256_loadu_pd(&a[i]), _mm256_loadu_pd(&b[i])));
    for (size_t i = (n & ~3); i < n; ++i)
        dst[i] = a[i] / b[i];
}

// Accumulate: dst[l] += a[l]   for l in [0, n)
inline void vadd_eq4(double* __restrict dst, const double* __restrict a,
                     size_t n) noexcept {
    for (size_t i = 0; i + 4 <= n; i += 4)
        _mm256_storeu_pd(&dst[i],
            _mm256_add_pd(_mm256_loadu_pd(&dst[i]), _mm256_loadu_pd(&a[i])));
    for (size_t i = (n & ~3); i < n; ++i)
        dst[i] += a[i];
}

inline void vsub_eq4(double* __restrict dst, const double* __restrict a,
                     size_t n) noexcept {
    for (size_t i = 0; i + 4 <= n; i += 4)
        _mm256_storeu_pd(&dst[i],
            _mm256_sub_pd(_mm256_loadu_pd(&dst[i]), _mm256_loadu_pd(&a[i])));
    for (size_t i = (n & ~3); i < n; ++i)
        dst[i] -= a[i];
}

// Scale-accumulate: dst[l] += s * a[l]   for l in [0, n)
inline void vmscadd4(double* __restrict dst, const double* __restrict a,
                     size_t n, double s) noexcept {
    __m256d sv = _mm256_set1_pd(s);
    for (size_t i = 0; i + 4 <= n; i += 4)
        _mm256_storeu_pd(&dst[i],
            _mm256_add_pd(_mm256_loadu_pd(&dst[i]),
                          _mm256_mul_pd(sv, _mm256_loadu_pd(&a[i]))));
    for (size_t i = (n & ~3); i < n; ++i)
        dst[i] += s * a[i];
}

// Two-term accumulate: dst[l] += a[l] * sc1 + b[l] * sc2   for l in [0, n)
inline void vtwosum4(double* __restrict dst,
                     const double* __restrict a,
                     const double* __restrict b,
                     size_t n,
                     double sc1, double sc2) noexcept {
    __m256d s1v = _mm256_set1_pd(sc1);
    __m256d s2v = _mm256_set1_pd(sc2);
    for (size_t i = 0; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d res = _mm256_add_pd(_mm256_mul_pd(s1v, va), _mm256_mul_pd(s2v, vb));
        _mm256_storeu_pd(&dst[i], _mm256_add_pd(_mm256_loadu_pd(&dst[i]), res));
    }
    for (size_t i = (n & ~3); i < n; ++i)
        dst[i] += a[i] * sc1 + b[i] * sc2;
}

// Copy: dst[l] = src[l]   for l in [0, n)
inline void vcopy4(double* __restrict dst, const double* __restrict src,
                   size_t n) noexcept {
    for (size_t i = 0; i + 4 <= n; i += 4)
        _mm256_storeu_pd(&dst[i], _mm256_loadu_pd(&src[i]));
    for (size_t i = (n & ~3); i < n; ++i)
        dst[i] = src[i];
}

// =====================================================================
// AVX-512 helpers (8 doubles @ 512-bit) — fall through to AVX2
// =====================================================================

// Only use AVX-512 if the compiler supports it AND we're not targeting
// a CPU that may lack it (e.g., Intel AMX / AVX512F)
#if defined(__AVX512F__) && !defined(AD_DISABLE_AVX512)
#define AD_HAS_AVX512 1

inline void vfill_n_avx512(double* __restrict dst, size_t n) noexcept {
    __m512d z = _mm512_setzero_pd();
    for (size_t i = 0; i + 8 <= n; i += 8)
        _mm512_storeu_pd(&dst[i], z);
    for (size_t i = (n & ~7); i < n; ++i)
        dst[i] = 0.0;
}

inline void vscal4_avx512(double* __restrict dst, const double* __restrict src,
                          size_t n, double coeff) noexcept {
    __m512d c = _mm512_set1_pd(coeff);
    for (size_t i = 0; i + 8 <= n; i += 8)
        _mm512_storeu_pd(&dst[i],
            _mm512_mul_pd(_mm512_loadu_pd(&src[i]), c));
    for (size_t i = (n & ~7); i < n; ++i)
        dst[i] = coeff * src[i];
}

inline void vadd4_avx512(double* __restrict dst, const double* __restrict a,
                         const double* __restrict b, size_t n) noexcept {
    for (size_t i = 0; i + 8 <= n; i += 8)
        _mm512_storeu_pd(&dst[i],
            _mm512_add_pd(_mm512_loadu_pd(&a[i]), _mm512_loadu_pd(&b[i])));
    for (size_t i = (n & ~7); i < n; ++i)
        dst[i] = a[i] + b[i];
}

inline void vsub4_avx512(double* __restrict dst, const double* __restrict a,
                         const double* __restrict b, size_t n) noexcept {
    for (size_t i = 0; i + 8 <= n; i += 8)
        _mm512_storeu_pd(&dst[i],
            _mm512_sub_pd(_mm512_loadu_pd(&a[i]), _mm512_loadu_pd(&b[i])));
    for (size_t i = (n & ~7); i < n; ++i)
        dst[i] = a[i] - b[i];
}

inline void vmul4_avx512(double* __restrict dst, const double* __restrict a,
                         const double* __restrict b, size_t n) noexcept {
    for (size_t i = 0; i + 8 <= n; i += 8)
        _mm512_storeu_pd(&dst[i],
            _mm512_mul_pd(_mm512_loadu_pd(&a[i]), _mm512_loadu_pd(&b[i])));
    for (size_t i = (n & ~7); i < n; ++i)
        dst[i] = a[i] * b[i];
}

inline void vdiv4_avx512(double* __restrict dst, const double* __restrict a,
                         const double* __restrict b, size_t n) noexcept {
    for (size_t i = 0; i + 8 <= n; i += 8)
        _mm512_storeu_pd(&dst[i],
            _mm512_div_pd(_mm512_loadu_pd(&a[i]), _mm512_loadu_pd(&b[i])));
    for (size_t i = (n & ~7); i < n; ++i)
        dst[i] = a[i] / b[i];
}

inline void vadd_eq4_avx512(double* __restrict dst, const double* __restrict a,
                            size_t n) noexcept {
    for (size_t i = 0; i + 8 <= n; i += 8)
        _mm512_storeu_pd(&dst[i],
            _mm512_add_pd(_mm512_loadu_pd(&dst[i]), _mm512_loadu_pd(&a[i])));
    for (size_t i = (n & ~7); i < n; ++i)
        dst[i] += a[i];
}

inline void vsub_eq4_avx512(double* __restrict dst, const double* __restrict a,
                            size_t n) noexcept {
    for (size_t i = 0; i + 8 <= n; i += 8)
        _mm512_storeu_pd(&dst[i],
            _mm512_sub_pd(_mm512_loadu_pd(&dst[i]), _mm512_loadu_pd(&a[i])));
    for (size_t i = (n & ~7); i < n; ++i)
        dst[i] -= a[i];
}

inline void vmscadd4_avx512(double* __restrict dst, const double* __restrict a,
                            size_t n, double s) noexcept {
    __m512d sv = _mm512_set1_pd(s);
    for (size_t i = 0; i + 8 <= n; i += 8)
        _mm512_storeu_pd(&dst[i],
            _mm512_add_pd(_mm512_loadu_pd(&dst[i]),
                          _mm512_mul_pd(sv, _mm512_loadu_pd(&a[i]))));
    for (size_t i = (n & ~7); i < n; ++i)
        dst[i] += s * a[i];
}

inline void vtwosum4_avx512(double* __restrict dst,
                            const double* __restrict a,
                            const double* __restrict b,
                            size_t n,
                            double sc1, double sc2) noexcept {
    __m512d s1v = _mm512_set1_pd(sc1);
    __m512d s2v = _mm512_set1_pd(sc2);
    for (size_t i = 0; i + 8 <= n; i += 8) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);
        __m512d res = _mm512_add_pd(_mm512_mul_pd(s1v, va), _mm512_mul_pd(s2v, vb));
        _mm512_storeu_pd(&dst[i], _mm512_add_pd(_mm512_loadu_pd(&dst[i]), res));
    }
    for (size_t i = (n & ~7); i < n; ++i)
        dst[i] += a[i] * sc1 + b[i] * sc2;
}

inline void vcopy4_avx512(double* __restrict dst, const double* __restrict src,
                          size_t n) noexcept {
    for (size_t i = 0; i + 8 <= n; i += 8)
        _mm512_storeu_pd(&dst[i], _mm512_loadu_pd(&src[i]));
    for (size_t i = (n & ~7); i < n; ++i)
        dst[i] = src[i];
}

#else  // no AVX-512
// Redirect to AVX2 for n <= 4, scalar for remainder
inline void vfill_n_avx512(double* __restrict dst, size_t n) noexcept {
    vfill_n(dst, n);
}
inline void vscal4_avx512(double* __restrict dst, const double* __restrict src,
                          size_t n, double coeff) noexcept {
    vscal4(dst, src, n, coeff);
}
inline void vadd4_avx512(double* __restrict dst, const double* __restrict a,
                         const double* __restrict b, size_t n) noexcept {
    vadd4(dst, a, b, n);
}
inline void vsub4_avx512(double* __restrict dst, const double* __restrict a,
                         const double* __restrict b, size_t n) noexcept {
    vsub4(dst, a, b, n);
}
inline void vmul4_avx512(double* __restrict dst, const double* __restrict a,
                         const double* __restrict b, size_t n) noexcept {
    vmul4(dst, a, b, n);
}
inline void vdiv4_avx512(double* __restrict dst, const double* __restrict a,
                         const double* __restrict b, size_t n) noexcept {
    vdiv4(dst, a, b, n);
}
inline void vadd_eq4_avx512(double* __restrict dst, const double* __restrict a,
                            size_t n) noexcept {
    vadd_eq4(dst, a, n);
}
inline void vsub_eq4_avx512(double* __restrict dst, const double* __restrict a,
                            size_t n) noexcept {
    vsub_eq4(dst, a, n);
}
inline void vmscadd4_avx512(double* __restrict dst, const double* __restrict a,
                            size_t n, double s) noexcept {
    vmscadd4(dst, a, n, s);
}
inline void vtwosum4_avx512(double* __restrict dst,
                            const double* __restrict a,
                            const double* __restrict b,
                            size_t n,
                            double sc1, double sc2) noexcept {
    vtwosum4(dst, a, b, n, sc1, sc2);
}
inline void vcopy4_avx512(double* __restrict dst, const double* __restrict src,
                          size_t n) noexcept {
    vcopy4(dst, src, n);
}
#endif // AVX-512

} // namespace ad::simd

#else // !defined(__AVX2__) && !defined(AD_DISABLE_SIMD)
#define AD_HAS_SIMD 0

// =====================================================================
// Non-SIMD fallback (scalar)
// =====================================================================
namespace ad::simd {

inline void vfill_n(double* __restrict dst, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) dst[i] = 0.0;
}
inline void vfill_n_val(double* __restrict dst, size_t n, double val) noexcept {
    for (size_t i = 0; i < n; ++i) dst[i] = val;
}
inline void vscal4(double* __restrict dst, const double* __restrict src,
                   size_t n, double coeff) noexcept {
    for (size_t i = 0; i < n; ++i) dst[i] = coeff * src[i];
}
inline void vadd4(double* __restrict dst, const double* __restrict a,
                  const double* __restrict b, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) dst[i] = a[i] + b[i];
}
inline void vsub4(double* __restrict dst, const double* __restrict a,
                  const double* __restrict b, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) dst[i] = a[i] - b[i];
}
inline void vmul4(double* __restrict dst, const double* __restrict a,
                  const double* __restrict b, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) dst[i] = a[i] * b[i];
}
inline void vdiv4(double* __restrict dst, const double* __restrict a,
                  const double* __restrict b, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) dst[i] = a[i] / b[i];
}
inline void vadd_eq4(double* __restrict dst, const double* __restrict a,
                     size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) dst[i] += a[i];
}
inline void vsub_eq4(double* __restrict dst, const double* __restrict a,
                     size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) dst[i] -= a[i];
}
inline void vmscadd4(double* __restrict dst, const double* __restrict a,
                     size_t n, double s) noexcept {
    for (size_t i = 0; i < n; ++i) dst[i] += s * a[i];
}
inline void vtwosum4(double* __restrict dst,
                     const double* __restrict a,
                     const double* __restrict b,
                     size_t n,
                     double sc1, double sc2) noexcept {
    for (size_t i = 0; i < n; ++i) dst[i] += a[i] * sc1 + b[i] * sc2;
}
inline void vcopy4(double* __restrict dst, const double* __restrict src,
                   size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) dst[i] = src[i];
}
// AVX-512 stubs redirect to scalar
inline void vfill_n_avx512(double* __restrict dst, size_t n) noexcept { vfill_n(dst, n); }
inline void vscal4_avx512(double* __restrict dst, const double* __restrict src,
                          size_t n, double coeff) noexcept { vscal4(dst, src, n, coeff); }
inline void vadd4_avx512(double* __restrict dst, const double* __restrict a,
                         const double* __restrict b, size_t n) noexcept { vadd4(dst, a, b, n); }
inline void vsub4_avx512(double* __restrict dst, const double* __restrict a,
                         const double* __restrict b, size_t n) noexcept { vsub4(dst, a, b, n); }
inline void vmul4_avx512(double* __restrict dst, const double* __restrict a,
                         const double* __restrict b, size_t n) noexcept { vmul4(dst, a, b, n); }
inline void vdiv4_avx512(double* __restrict dst, const double* __restrict a,
                         const double* __restrict b, size_t n) noexcept { vdiv4(dst, a, b, n); }
inline void vadd_eq4_avx512(double* __restrict dst, const double* __restrict a,
                            size_t n) noexcept { vadd_eq4(dst, a, n); }
inline void vsub_eq4_avx512(double* __restrict dst, const double* __restrict a,
                            size_t n) noexcept { vsub_eq4(dst, a, n); }
inline void vmscadd4_avx512(double* __restrict dst, const double* __restrict a,
                            size_t n, double s) noexcept { vmscadd4(dst, a, n, s); }
inline void vtwosum4_avx512(double* __restrict dst,
                            const double* __restrict a,
                            const double* __restrict b,
                            size_t n,
                            double sc1, double sc2) noexcept {
    vtwosum4(dst, a, b, n, sc1, sc2);
}
inline void vcopy4_avx512(double* __restrict dst, const double* __restrict src,
                          size_t n) noexcept { vcopy4(dst, src, n); }

} // namespace ad::simd

#endif // AD_HAS_SIMD
