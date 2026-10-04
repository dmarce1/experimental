# CosmicTiger

C++20 project using HPX, MPI, CUDA, Silo, and GoogleTest.

Project layout:

- `src/`: C++ source files.
- `cosmictiger/`: public C++ headers, included as `<cosmictiger/name.hpp>`.
- `tests/`: GoogleTest C++ source files.

## Build and run

Requires CMake 3.25+, a C++20 compiler, MPI, HPX built with networking and its
MPI parcelport enabled, pkg-config, Silo, and the CUDA toolkit.
Set `CMAKE_PREFIX_PATH` to your HPX installation when needed.
GoogleTest is found locally or downloaded at a pinned revision during configure.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$HOME/local/hpx/Release"
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure --parallel 2
./build/cosmictiger --hpx:threads=4
mpiexec -n 2 ./build/cosmictiger --hpx:threads=2 --hpx:bind=none \
    --hpx:ini=hpx.parcel.mpi.enable=1 --hpx:ini=hpx.parcel.tcp.enable=0
```

The executable starts the HPX runtime, prints the startup screen on the console
locality, and exits. Use `mpiexec` to start multiple MPI localities.
Choose the HPX worker count explicitly to respect local resource limits.

CUDA is enabled by default. Both C++ and CUDA sources use C++20.
Add CUDA sources to the appropriate CMake
target and set `CMAKE_CUDA_ARCHITECTURES` for the GPUs you intend to support.
For a CPU-only build, pass `-DCOSMICTIGER_ENABLE_CUDA=OFF`.
To omit the test suite, pass `-DBUILD_TESTING=OFF`.
