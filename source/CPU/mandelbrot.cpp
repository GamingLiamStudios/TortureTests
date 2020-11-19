#include "mandelbrot.h"

#include "stb_image_write.h"
#include <iostream>
#include <complex>
#include <stdint.h>
#include "fastmath.h"
#include "v2d.h"
#include <immintrin.h>
#include <cmath>

void CPU::Mandelbrot(int width, int height, int iterations)
{
    uint8_t *image = (uint8_t *) malloc(width * height * 3);

    double zoom = pow(1.1, 2);

    v2df_d fr_tl = { (2.0 / 1.1) * zoom - .1, 1.0 * zoom },
           fr_br = { (-.5 / 1.1) * zoom - .1, -1.0 * zoom };

    double x_scale = (fr_br.x - fr_tl.x) / double(width);
    double y_scale = (fr_br.y - fr_tl.y) / double(height);

    double y_pos = fr_tl.y;

    int y_offset = 0;
    int row_size = width;

    int x, y;

#ifndef __AVX__

    double x_pos = 0;

    float  n, a = 0.1f, lv;
    int    nf;
    __m128 sinv, interpv;

    std::complex<double> c, z;

    for (y = 0; y < height; y++)
    {
        x_pos = 0;
        for (x = 0; x < width; x++)
        {
            c = { x_pos + -2.0, y_pos + -1.0 };
            z = { 0, 0 };

            n = 0;
            while (abs(z) < 2.0 && n < iterations)
            {
                z = (z * z) + c;
                n++;
            }

            if (n < iterations)
            {
                lv = flogf(abs(z)) / 2;
                lv = flogf(lv / flogf(2)) / flogf(2);
                n  = n + 1 - lv;
            }
            nf = floorf(n);

            {    // Visualize Calculated point
                // 0.5f * fsin(a * n + ...) + 0.5f
                sinv = _mm_set_ps(0, 2.094f, 4.188f, 0);
                sinv = _mm_add_ps(_mm_set1_ps(a * nf), sinv);
                sinv = _mm_ext_fsin_ps(sinv);
                sinv = _mm_mul_ps(_mm_set1_ps(0.5f), sinv);
                sinv = _mm_add_ps(sinv, _mm_set1_ps(0.5f));

                nf++;
                // 0.5f * fsin(a * n + ...) + 0.5f
                interpv = _mm_set_ps(0, 2.094f, 4.188f, 0);
                interpv = _mm_add_ps(_mm_set1_ps(a * nf), interpv);
                interpv = _mm_ext_fsin_ps(interpv);
                interpv = _mm_mul_ps(_mm_set1_ps(0.5f), interpv);
                interpv = _mm_add_ps(interpv, _mm_set1_ps(0.5f));

                // Interpolate values
                n       = n - (int) n;
                sinv    = _mm_mul_ps(_mm_set1_ps(1 - n), sinv);
                interpv = _mm_mul_ps(_mm_set1_ps(n), interpv);
                sinv    = _mm_add_ps(sinv, interpv);

                // Write to image buffer
                sinv                          = _mm_mul_ps(sinv, _mm_set1_ps(255.0f));
                image[(y_offset + x) * 3 + 0] = ((float *) &sinv)[0];
                image[(y_offset + x) * 3 + 1] = ((float *) &sinv)[1];
                image[(y_offset + x) * 3 + 2] = ((float *) &sinv)[2];
            }

            x_pos += x_scale;
        }

        y_pos += y_scale;
        y_offset += row_size;
    }

#else

    __m256d x_pos, a, b, two, escr2;
    __m256d zr, zi, re, im, ci;
    __m256i ni, c, mask, itt, one;
    __m256  sinv, d;
    __m128  log, mask1;

    float  pa = 0.1f;
    double nv;
    int    i, temp;

    one   = _mm256_set1_epi64x(1);
    two   = _mm256_set1_pd(2.0);
    escr2 = _mm256_set1_pd(65536.0);
    itt   = _mm256_set1_epi64x(iterations);

    for (y = 0; y < height; y++)
    {
        // Set x_pos to 'zero' state
        x_pos = _mm256_setr_pd(0, 1, 2, 3);
        x_pos = _mm256_mul_pd(x_pos, _mm256_set1_pd(x_scale));
        x_pos = _mm256_add_pd(_mm256_set1_pd(fr_tl.x), x_pos);

        ci = _mm256_set1_pd(y_pos);

        for (x = 0; x < width; x += 4)    // += 4 because of AVX doing 4 at a time
        {
            // Initalize values
            zr = zi = _mm256_setzero_pd();
            ni      = _mm256_setzero_si256();

repeat:
            // z = (z * z) + c;
            a = _mm256_mul_pd(zr, zr);
            im = _mm256_mul_pd(zr, zi);
#if defined(__AVX2__) || defined(__FMA__)
            re = _mm256_fmadd_pd(zi, zi, x_pos);
            b = _mm256_fmadd_pd(zi, zi, a);
            zi = _mm256_fmadd_pd(two, im, ci);
#else
            zr = _mm256_mul_pd(zi, zi);
            re = _mm256_add_pd(zr, x_pos);
            b  = _mm256_mul_pd(two, im);
            zi = _mm256_add_pd(b, ci);
            b  = _mm256_add_pd(zr, a);
#endif
            zr = _mm256_sub_pd(a, re);

            // While logic
            // TODO: Modify for AVX compat
            a = _mm256_cmp_pd(b, escr2, _CMP_LE_OQ);
            mask = _mm256_cmpgt_epi64(itt, ni);
            mask = _mm256_and_si256(mask, _mm256_castpd_si256(a));
            c = _mm256_and_si256(one, mask);
            ni = _mm256_add_epi64(ni, c);
            if (_mm256_movemask_pd(_mm256_castsi256_pd(mask)) > 0) goto repeat;

            a = _mm256_sqrt_pd(b);
            b = _mm256_ext_flog_pd(a);
            a = _mm256_ext_flog_pd(b);
            // TODO: Precalculate log(2)
            b = _mm256_div_pd(a, _mm256_ext_flog_pd(_mm256_set1_pd(2.0)));
            a = _mm256_setr_pd(
              _mm256_extract_epi64(ni, 0) + 1,
              _mm256_extract_epi64(ni, 1) + 1,
              _mm256_extract_epi64(ni, 2) + 1,
              _mm256_extract_epi64(ni, 3) + 1);
            a = _mm256_sub_pd(a, b);

            for (i = 0; i < 4; i++)
            {    // Visualize Calculated points
                // 0.5f * fsin(a * n + ...) + 0.5f
                nv = ((double *) &a)[i];
                std::cout << std::to_string(nv - ((long *) &ni)[i]) << "\n";
                sinv = _mm256_set_ps(0, 2.094f, 4.188f, 0, 0, 2.094f, 4.188f, 0);
                d = _mm256_castps128_ps256(_mm_set1_ps(pa * floor(nv)));
                d = _mm256_insertf128_ps(d, _mm_set1_ps(pa * floor(nv) + 1), 1);
                sinv = _mm256_add_ps(d, sinv);
                sinv = _mm256_ext_fsin_ps(sinv);
                sinv = _mm256_mul_ps(_mm256_set1_ps(0.5f), sinv);
                sinv = _mm256_add_ps(sinv, _mm256_set1_ps(0.5f));
                // Convert to 0-255 RGB
                sinv = _mm256_mul_ps(sinv, _mm256_set1_ps(255.0f));

                // Interpolate values
                nv = abs(nv - (long) nv);
                log = _mm_sub_ps(((__m128 *) &sinv)[1], ((__m128 *) &sinv)[0]);
                // TODO: '#ifdef' Check for compatability
                mask1 = _mm_fmadd_ps(_mm_set1_ps(nv), log, ((__m128 *) &sinv)[0]);

                _MM_EXTRACT_FLOAT(image[(y_offset + x + i) * 3 + 0], mask1, 0);
                _MM_EXTRACT_FLOAT(image[(y_offset + x + i) * 3 + 1], mask1, 1);
                _MM_EXTRACT_FLOAT(image[(y_offset + x + i) * 3 + 2], mask1, 2);
            }

            x_pos = _mm256_add_pd(x_pos, _mm256_set1_pd(x_scale * 4));
        }

        y_pos += y_scale;
        y_offset += row_size;
    }

#endif

    stbi_write_png("result.png", width, height, 3, image, width * 3);
    free(image);
}