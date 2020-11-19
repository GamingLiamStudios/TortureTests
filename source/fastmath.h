#pragma once

#include <immintrin.h>

static inline float fsinf(float x)
{
    static const float fouroverpi   = 1.2732395447351627f;
    static const float fouroverpisq = 0.40528473456935109f;
    static const float q            = 0.77633023248007499f;
    union
    {
        float    f;
        uint32_t i;
    } p = { 0.22308510060189463f };

    union
    {
        float    f;
        uint32_t i;
    } vx          = { x };
    uint32_t sign = vx.i & 0x80000000;
    vx.i &= 0x7FFFFFFF;

    float qpprox = fouroverpi * x - fouroverpisq * x * vx.f;

    p.i |= sign;

    return qpprox * (q + p.f * qpprox);
}

static inline float flogf(float x)
{
    union
    {
        float    f;
        uint32_t i;
    } xv = { x }, lv;
    lv.i = 0x43800000u | (xv.i >> 8u);
    return 0.69314718f * (lv.f - 382.95695f);
}

#ifdef __SSE2__

// SSE implementation of fsinf
static inline __m128 _mm_ext_fsin_ps(const __m128 x)
{
    // Initalize variables
    const __m128 fouroverpi   = _mm_set1_ps(1.2732395447351627f);
    const __m128 fouroverpisq = _mm_set1_ps(0.40528473456935109f);
    const __m128 q            = _mm_set1_ps(0.77633023248007499f);
    union
    {
        __m128  f;
        __m128i i;
    } p;

    union
    {
        __m128  f;
        __m128i i;
    } vx         = { x };
    __m128i sign = _mm_and_si128(vx.i, _mm_set1_epi32(0x80000000));
    vx.i         = _mm_and_si128(vx.i, _mm_set1_epi32(0x7FFFFFFF));

    __m128 qpprox = _mm_mul_ps(fouroverpi, x);
    p.f           = _mm_mul_ps(fouroverpisq, x);
    p.f           = _mm_mul_ps(p.f, vx.f);
    qpprox        = _mm_sub_ps(qpprox, p.f);

    p.f = _mm_set1_ps(0.22308510060189463f);
    p.i = _mm_or_si128(p.i, sign);

    vx.f = _mm_mul_ps(p.f, qpprox);
    vx.f = _mm_add_ps(q, vx.f);
    return _mm_mul_ps(qpprox, vx.f);
}

// SSE implementation of flogf
static inline __m128 _mm_ext_flog_ps(__m128 x)
{
    union
    {
        __m128  f;
        __m128i i;
    } xv = { x }, lv;
    lv.i = _mm_set1_epi32(0x43800000u);

    xv.i = _mm_srli_epi32(xv.i, 8);
    lv.i = _mm_or_si128(lv.i, xv.i);

    xv.f = _mm_set1_ps(382.95695f);
    lv.f = _mm_sub_ps(lv.f, xv.f);
    xv.f = _mm_set1_ps(0.69314718f);
    return _mm_mul_ps(xv.f, lv.f);
}

#endif

#ifdef __AVX__

// AVX implementation of fsinf
static inline __m256 _mm256_ext_fsin_ps(const __m256 x)
{
    // Initalize variables
    const __m256 fouroverpi   = _mm256_set1_ps(1.2732395447351627f);
    const __m256 fouroverpisq = _mm256_set1_ps(0.40528473456935109f);
    const __m256 q            = _mm256_set1_ps(0.77633023248007499f);
    union
    {
        __m256  f;
        __m256i i;
    } p;

    union
    {
        __m256  f;
        __m256i i;
    } vx         = { x };
    __m256i sign = _mm256_and_si256(vx.i, _mm256_set1_epi32(0x80000000));
    vx.i         = _mm256_and_si256(vx.i, _mm256_set1_epi32(0x7FFFFFFF));

    p.f = _mm256_mul_ps(fouroverpisq, x);
    p.f = _mm256_mul_ps(p.f, vx.f);
#if defined(__AVX2__) || defined(__FMA__)
    __m256 qpprox = _mm256_fmsub_ps(fouroverpi, x, p.f);
#else
    __m256 qpprox = _mm256_mul_ps(fouroverpi, x);
    qpprox        = _mm256_sub_ps(qpprox, p.f);
#endif

    p.f = _mm256_set1_ps(0.22308510060189463f);
    p.i = _mm256_or_si256(p.i, sign);

    vx.f = _mm256_mul_ps(p.f, qpprox);
    vx.f = _mm256_add_ps(q, vx.f);
    return _mm256_mul_ps(qpprox, vx.f);
}

// AVX implementation of flogf
static inline __m256 _mm256_ext_flog_ps(__m256 x)
{
    union
    {
        __m256  f;
        __m256i i;
    } xv = { x }, lv;
    lv.i = _mm256_set1_epi32(0x43800000u);

    xv.i = _mm256_srli_epi32(xv.i, 8);
    lv.i = _mm256_or_epi32(lv.i, xv.i);

    xv.f = _mm256_set1_ps(382.95695f);
    lv.f = _mm256_sub_ps(lv.f, xv.f);
    xv.f = _mm256_set1_ps(0.69314718f);
    return _mm256_mul_ps(xv.f, lv.f);
}

static inline __m256d _mm256_ext_flog_pd(__m256d x)
{
    union
    {
        __m256d f;
        __m256i i;
    } xv = { x }, lv;
    lv.i = _mm256_set1_epi64x(0x43800000u);

    xv.i = _mm256_srli_epi64(xv.i, 8);
    lv.i = _mm256_or_si256(lv.i, xv.i);

    xv.f = _mm256_set1_pd(382.95695f);
    lv.f = _mm256_sub_pd(lv.f, xv.f);
    xv.f = _mm256_set1_pd(0.69314718f);
    return _mm256_mul_pd(xv.f, lv.f);
}

static inline __m256d _mm256_ext_fln_pd(__m256d x)
{
    return _mm256_div_pd(
      _mm256_ext_flog_pd(x),
      _mm256_ext_flog_pd(_mm256_set1_pd(2.7182818284591)));
}

#endif