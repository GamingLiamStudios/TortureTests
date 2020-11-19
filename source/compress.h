#pragma once

#include "zlib-ng.h"
#include "malloc.h"

unsigned char *compress(unsigned char *data, int data_len, int *out_len, int quality)
{
    size_t         size = zng_compressBound(data_len);
    unsigned char *out  = (unsigned char *) malloc(size);

    zng_compress2(out, &size, data, data_len, quality);

    out      = (unsigned char *) realloc(out, size);
    *out_len = size;
    return out;
}