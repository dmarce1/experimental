#include <cosmictiger/startup.hpp>

#include <hpx/hpx_init.hpp>
#include <silo.h>

#ifdef COSMICTIGER_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

#include <iostream>

static_assert(__cplusplus >= 202002L, "CosmicTiger requires C++20.");

int hpx_main() {
    std::cout << cosmictiger::startup_screen
              << "  Silo version: " << DBVersion() << '\n';
#ifdef COSMICTIGER_ENABLE_CUDA
    std::cout << "  CUDA toolkit: " << CUDART_VERSION / 1000 << '.'
              << (CUDART_VERSION % 1000) / 10 << '\n';
#endif
    std::cout << "  HPX runtime started with MPI support.\n\n";
    return hpx::finalize();
}

int main(int argc, char* argv[]) {
    return hpx::init(argc, argv);
}
