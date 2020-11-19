// Library pre-defines
#include "compress.h"    // Do I need this? no. Will it make it faster? yes
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBIW_ZLIB_COMPRESS compress
#include "stb_image_write.h"

#define OLC_IMAGE_STB
#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

// Util for main()

#include <iostream>
#include <chrono>

// CPU Torture Tests

#include "CPU/mandelbrot.h"

int main()
{
    /*
    std::cout << "Welcome to \"Fuck my Computer\"\nPlease select your murder weapon:\n";
    std::cout << "1. Mandelbrot CPU\n";

    std::cout << "\n";

    // Read selection
    std::string in;
    in.resize(12);
    std::cin.getline(in.data(), 12);
    int selector = std::stoi(in);

    switch (selector)
    {
    case 1: CPU::Mandelbrot(7680, 4320); break;
    default: std::cout << "Invalid Option";
    }
    std::cin.get();
    */

    auto last = std::chrono::high_resolution_clock::now();

    CPU::Mandelbrot(4096, 4096, 1 << 10);

    auto duration = std::chrono::high_resolution_clock::now() - last;

    std::cout << std::to_string(duration.count() / 1000000000.0);
    return 0;
}