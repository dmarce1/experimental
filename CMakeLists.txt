# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026  Dominic C. Marcello
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

cmake_minimum_required(VERSION 3.25)
project(cosmictiger VERSION 0.1.0 LANGUAGES CXX)

if(CMAKE_CONFIGURATION_TYPES)
    message(FATAL_ERROR "Use a single-configuration generator and a separate build directory per mode: HPX APEX support differs between modes.")
endif()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "CosmicTiger build type" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release RelWithDebInfo Debug Profile)
if(NOT CMAKE_BUILD_TYPE MATCHES "^(Release|RelWithDebInfo|Debug|Profile)$")
    message(FATAL_ERROR "Choose CMAKE_BUILD_TYPE=Release, RelWithDebInfo, Debug, or Profile.")
endif()
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    message(FATAL_ERROR "CosmicTiger build flags require a GCC-compatible or Clang-compatible compiler.")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native -g0 -DNDEBUG" CACHE STRING "Release C++ flags" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-Ofast -march=native -g -DNDEBUG" CACHE STRING "RelWithDebInfo C++ flags" FORCE)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -march=native -g3 -UNDEBUG" CACHE STRING "Debug C++ flags" FORCE)
# CMake applies these language flags at both compilation and linking, as required for -pg.
set(CMAKE_CXX_FLAGS_PROFILE "-O2 -pg -g -march=native -DNDEBUG" CACHE STRING "Profile C++ flags" FORCE)
# The empty entry permits dependencies such as CUDA::cudart with no configuration suffix.
set(CMAKE_MAP_IMPORTED_CONFIG_PROFILE RelWithDebInfo Release "")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(COSMICTIGER_ENABLE_CUDA "Enable CUDA compilation and runtime support" ON)
option(COSMICTIGER_ENABLE_PROFILING "Enable statistical CPU sampling and PROFILE() metadata" ON)
if(NOT COSMICTIGER_ENABLE_PROFILING)
    add_compile_definitions(PROFILE_DISABLE=1)
endif()

# Numerical contracts, including particle bounds, are checked in every mode
# except Release. Keep IEEE exception semantics in validation-enabled builds.
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(COSMICTIGER_VALIDATION_CHECKS=1)
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math;-ftrapping-math>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-fast-math,-ftrapping-math>")
    add_link_options(
        "$<$<LINK_LANGUAGE:CXX>:-fno-fast-math;-ftrapping-math>"
        "$<$<LINK_LANGUAGE:CUDA>:-Xcompiler=-fno-fast-math,-ftrapping-math>")
    # GCC's -Ofast driver otherwise links crtfastmath.o, which enables
    # flush-to-zero even when -fno-fast-math was used while compiling.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|i.86)$")
        add_link_options(
            "$<$<LINK_LANGUAGE:CXX>:-mno-daz-ftz>"
            "$<$<LINK_LANGUAGE:CUDA>:-Xcompiler=-mno-daz-ftz>")
    endif()
endif()

# Every executable supports runtime FMM orders 2 through 9. Retire the
# former cache choices so existing build directories gain the full range.
unset(COSMICTIGER_MULTIPOLE_ORDER CACHE)
unset(COSMICTIGER_EXPANSION_ORDER CACHE)
set(COSMICTIGER_MULTIPOLE_ORDER 8)
set(COSMICTIGER_EXPANSION_ORDER 9)

find_package(HPX CONFIG REQUIRED)
add_library(cosmictiger_profiler STATIC src/profiler.cpp)
target_include_directories(cosmictiger_profiler PUBLIC "${PROJECT_SOURCE_DIR}")
target_compile_features(cosmictiger_profiler PUBLIC cxx_std_20)
# Keep this target independent of application/header libraries: those libraries
# depend on it when their public headers contain PROFILE() registration calls.
target_link_libraries(cosmictiger_profiler PUBLIC HPX::hpx ${CMAKE_DL_LIBS} "$<$<PLATFORM_ID:Linux>:rt>")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(HPX_WITH_APEX)
        message(FATAL_ERROR "Release requires HPX built with HPX_WITH_APEX=OFF. Select that installation using HPX_DIR or CMAKE_PREFIX_PATH.")
    endif()
elseif(NOT HPX_WITH_APEX)
    message(FATAL_ERROR "${CMAKE_BUILD_TYPE} requires HPX built with HPX_WITH_APEX=ON. Select that installation using HPX_DIR or CMAKE_PREFIX_PATH.")
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
if(NOT HPX_WITH_NETWORKING OR NOT HPX_WITH_PARCELPORT_MPI)
    message(FATAL_ERROR "CosmicTiger requires HPX built with networking and the MPI parcelport enabled.")
endif()
find_package(PkgConfig REQUIRED)
pkg_check_modules(SILO REQUIRED IMPORTED_TARGET silo)
pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)

if(COSMICTIGER_ENABLE_CUDA)
    if(NOT HPX_WITH_CUDA)
        message(FATAL_ERROR "CUDA support requires HPX built with HPX_WITH_CUDA=ON. Select that installation using HPX_DIR or CMAKE_PREFIX_PATH.")
    endif()
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    enable_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        message(FATAL_ERROR "CosmicTiger CUDA build flags require the NVIDIA CUDA compiler.")
    endif()
    # NVCC uses its own optimization flags; forward the requested CPU flags to its host compiler.
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -Xcompiler=-Ofast,-march=native,-g0 -DNDEBUG" CACHE STRING "Release CUDA flags" FORCE)
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O3 -g -Xcompiler=-Ofast,-march=native,-g -DNDEBUG" CACHE STRING "RelWithDebInfo CUDA flags" FORCE)
    set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g -Xcompiler=-O0,-march=native,-g3,-UNDEBUG" CACHE STRING "Debug CUDA flags" FORCE)
    set(CMAKE_CUDA_FLAGS_PROFILE "-O2 -g -Xcompiler=-O2,-pg,-g,-march=native -DNDEBUG" CACHE STRING "Profile CUDA flags" FORCE)
    find_package(CUDAToolkit REQUIRED)
endif()

find_package(Boost REQUIRED)
add_library(cosmictiger_units INTERFACE)
target_include_directories(cosmictiger_units INTERFACE "${PROJECT_SOURCE_DIR}")
target_compile_features(cosmictiger_units INTERFACE cxx_std_20)
target_link_libraries(cosmictiger_units INTERFACE Boost::headers)

add_library(cosmictiger_dependencies INTERFACE)
target_include_directories(cosmictiger_dependencies INTERFACE "${PROJECT_SOURCE_DIR}")
target_compile_definitions(cosmictiger_dependencies INTERFACE
    COSMICTIGER_MULTIPOLE_ORDER=${COSMICTIGER_MULTIPOLE_ORDER}
    COSMICTIGER_EXPANSION_ORDER=${COSMICTIGER_EXPANSION_ORDER})
target_link_libraries(cosmictiger_dependencies INTERFACE HPX::hpx MPI::MPI_CXX PkgConfig::SILO PkgConfig::FFTW3 cosmictiger_units)
if(COSMICTIGER_ENABLE_CUDA)
    add_library(cosmictiger_cuda_runtime STATIC src/cuda/cuda_runtime.cpp)
    target_include_directories(cosmictiger_cuda_runtime PUBLIC "${PROJECT_SOURCE_DIR}")
    target_compile_features(cosmictiger_cuda_runtime PUBLIC cxx_std_20)
    target_link_libraries(cosmictiger_cuda_runtime PUBLIC HPX::hpx CUDA::cudart)
    target_compile_definitions(cosmictiger_cuda_runtime PUBLIC COSMICTIGER_ENABLE_CUDA=1)
    target_link_libraries(cosmictiger_dependencies INTERFACE cosmictiger_cuda_runtime)
    target_compile_definitions(cosmictiger_dependencies INTERFACE COSMICTIGER_ENABLE_CUDA=1)
endif()

add_library(cosmictiger_stf INTERFACE)
target_include_directories(cosmictiger_stf INTERFACE "${PROJECT_SOURCE_DIR}")
target_compile_features(cosmictiger_stf INTERFACE cxx_std_20)
if(COSMICTIGER_ENABLE_CUDA)
    add_library(cosmictiger_stf_m2p_cuda STATIC src/gravity/stf/m2p_cuda.cu src/gravity/symmetric/m2p_cuda.cu)
    target_link_libraries(cosmictiger_stf_m2p_cuda PUBLIC cosmictiger_dependencies)
endif()

# CAMB runs as a separate package. Only the C++ file/process adapter is linked.
include(cmake/camb.cmake)
add_library(cosmictiger_boltzmann STATIC src/einstein_boltzmann.cpp src/camb.cpp)
target_include_directories(cosmictiger_boltzmann PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(cosmictiger_boltzmann PUBLIC HPX::hpx)
target_compile_definitions(cosmictiger_boltzmann PRIVATE
    COSMICTIGER_CAMB_EXECUTABLE="${COSMICTIGER_CAMB_EXECUTABLE}")
# Keep input validation and spectrum interpolation IEEE-correct in -Ofast modes.
target_compile_options(cosmictiger_boltzmann PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic;-fno-fast-math>")
add_dependencies(cosmictiger_boltzmann cosmictiger_camb_solver)

add_library(cosmictiger_cosmology STATIC src/friedmann.cpp src/simulation_time.cpp src/evolution/evolution_end.cpp)
target_include_directories(cosmictiger_cosmology PUBLIC "${PROJECT_SOURCE_DIR}")
target_compile_features(cosmictiger_cosmology PUBLIC cxx_std_20)
target_link_libraries(cosmictiger_cosmology PUBLIC cosmictiger_units)
# This small background calculation needs IEEE finite-value guards and
# accurate cancellation, even in the project's -Ofast build modes.
target_compile_options(cosmictiger_cosmology PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic;-fno-fast-math>")

add_library(cosmictiger_fmm STATIC src/gravity/cartesian/fmm.cpp)
target_link_libraries(cosmictiger_fmm PUBLIC cosmictiger_stf HPX::hpx)
target_compile_options(cosmictiger_fmm PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic>")

# Batched cell-level FMM interactions (M2L, P2L, M2P), not yet attached to evolution.
add_library(cosmictiger_fmm_interactions STATIC src/gravity/fmm/fmm_interactions.cpp)
target_link_libraries(cosmictiger_fmm_interactions PUBLIC cosmictiger_dependencies)
target_compile_options(cosmictiger_fmm_interactions PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic>")
if(COSMICTIGER_ENABLE_CUDA)
    target_sources(cosmictiger_fmm_interactions PRIVATE src/gravity/fmm/fmm_interactions.cu)
endif()

add_library(cosmictiger_particles STATIC src/cartesian_grid.cpp src/particle/particle_load_balance.cpp src/particle/particle_ghost_exchange.cpp src/particle/particle_silo.cpp)
target_link_libraries(cosmictiger_particles PUBLIC cosmictiger_dependencies cosmictiger_cosmology)
target_compile_options(cosmictiger_particles PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic>")
if(COSMICTIGER_ENABLE_CUDA)
    target_sources(cosmictiger_particles PRIVATE src/gravity/p2m_cuda.cu)
endif()

# Per-cell kd-trees for local FMM solves; position ordering is the active path.
add_library(cosmictiger_kd_tree STATIC src/gravity/tree/kd_tree.cpp)
target_link_libraries(cosmictiger_kd_tree PUBLIC cosmictiger_particles)
target_compile_options(cosmictiger_kd_tree PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic>")

add_library(cosmictiger_initialization STATIC src/particle/particle_initialization.cpp src/fft/fft_hpx_exchange.cpp)
target_link_libraries(cosmictiger_initialization PUBLIC cosmictiger_particles cosmictiger_boltzmann)
# Threshold bracketing, finite-value guards, and CAMB amplitude derivatives
# require IEEE arithmetic in the application's -Ofast modes as well.
target_compile_options(cosmictiger_initialization PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic;-fno-fast-math>")

add_library(cosmictiger_pmmm STATIC src/gravity/pmmm/pmmm_runtime_convolution.cpp)
target_link_libraries(cosmictiger_pmmm PUBLIC cosmictiger_initialization)
target_compile_options(cosmictiger_pmmm PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic;-fno-fast-math>")

add_library(cosmictiger_healpix STATIC src/healpix/healpix.cpp src/healpix/healpix_simd.cpp)
target_include_directories(cosmictiger_healpix PUBLIC "${PROJECT_SOURCE_DIR}")
target_compile_features(cosmictiger_healpix PUBLIC cxx_std_20)
target_compile_options(cosmictiger_healpix PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-fno-fast-math;-ffp-contract=off>")

add_library(cosmictiger_lightcone STATIC src/lightcone/lightcone.cpp)
target_link_libraries(cosmictiger_lightcone PUBLIC cosmictiger_particles cosmictiger_cosmology cosmictiger_healpix)
target_compile_options(cosmictiger_lightcone PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-fno-fast-math>)

add_library(cosmictiger_global_constants STATIC src/global_constants.cpp src/command_line_options.cpp)
target_include_directories(cosmictiger_global_constants PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(cosmictiger_global_constants PUBLIC cosmictiger_initialization cosmictiger_cosmology HPX::hpx)
target_compile_options(cosmictiger_global_constants PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic;-fno-fast-math>")

add_library(cosmictiger_kick STATIC src/gravity/kick/kick.cpp src/gravity/kick/kick_sums.cpp src/gravity/kick/pmmm_kick.cpp)
target_link_libraries(cosmictiger_kick PUBLIC cosmictiger_global_constants cosmictiger_particles cosmictiger_fmm)
target_compile_options(cosmictiger_kick PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic>")
if(COSMICTIGER_ENABLE_CUDA)
    target_sources(cosmictiger_kick PRIVATE src/gravity/kick/pmmm_kick_cuda.cu)
endif()

add_library(cosmictiger_direct_solve STATIC src/gravity/direct/direct_solve.cpp src/gravity/direct/direct_comparison.cpp)
target_link_libraries(cosmictiger_direct_solve PUBLIC cosmictiger_global_constants cosmictiger_particles)
target_compile_options(cosmictiger_direct_solve PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic;-fno-fast-math>")
if(COSMICTIGER_ENABLE_CUDA)
    target_sources(cosmictiger_direct_solve PRIVATE src/gravity/direct/direct_solve.cu)
endif()

add_library(cosmictiger_drift STATIC src/drift/drift.cpp)
target_link_libraries(cosmictiger_drift PUBLIC cosmictiger_particles)
target_compile_options(cosmictiger_drift PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic>")
if(COSMICTIGER_ENABLE_CUDA)
    target_sources(cosmictiger_drift PRIVATE src/drift/drift_cuda.cu)
endif()

add_library(cosmictiger_evolution STATIC src/evolution/evolution.cpp src/evolution/evolution_state.cpp src/evolution/evolution_checkpoint.cpp)
target_link_libraries(cosmictiger_evolution PUBLIC cosmictiger_kick cosmictiger_drift cosmictiger_cosmology cosmictiger_lightcone cosmictiger_direct_solve cosmictiger_pmmm HPX::iostreams_component)
target_compile_options(cosmictiger_evolution PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic;-fno-fast-math>")

if(COSMICTIGER_ENABLE_CUDA)
    # The standalone allocator view and its HPX managed-memory host owner.
    add_library(cosmictiger_cuda_memory_pool STATIC src/cuda/cuda_memory_pool.cpp)
    target_include_directories(cosmictiger_cuda_memory_pool PUBLIC "${PROJECT_SOURCE_DIR}")
    target_compile_features(cosmictiger_cuda_memory_pool PUBLIC cxx_std_20)
    target_link_libraries(cosmictiger_cuda_memory_pool PUBLIC cosmictiger_cuda_runtime)

    add_library(cosmictiger_direct_cuda STATIC src/gravity/direct/direct_interactions.cu)
    target_link_libraries(cosmictiger_direct_cuda PUBLIC cosmictiger_dependencies)
endif()

add_executable(cosmictiger src/main.cpp)
target_include_directories(cosmictiger PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(cosmictiger PRIVATE HPX::hpx HPX::iostreams_component cosmictiger_cosmology cosmictiger_global_constants cosmictiger_kick)
target_link_libraries(cosmictiger PRIVATE cosmictiger_boltzmann cosmictiger_initialization cosmictiger_evolution)
target_compile_options(cosmictiger PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic>")

include(CTest)
if(BUILD_TESTING)
    if(COSMICTIGER_ENABLE_CUDA)
        add_executable(cosmictiger_cuda_memory_pool_host_tests tests/cuda/cuda_memory_pool_host.cpp)
        target_link_libraries(cosmictiger_cuda_memory_pool_host_tests PRIVATE cosmictiger_cuda_memory_pool HPX::hpx)
        add_test(NAME cosmictiger.cuda_memory_pool_host COMMAND cosmictiger_cuda_memory_pool_host_tests --hpx:threads=4)
        set_tests_properties(cosmictiger.cuda_memory_pool_host PROPERTIES PROCESSORS 4 TIMEOUT 120)

        # Generate Pascal code/PTX so the test can run on newer devices by JIT.
        add_executable(cosmictiger_cuda_memory_pool_cuda_tests tests/cuda/cuda_memory_pool_cuda.cu tests/cuda/cuda_memory_pool_main.cpp)
        set_property(TARGET cosmictiger_cuda_memory_pool_cuda_tests PROPERTY CUDA_ARCHITECTURES 60)
        target_link_libraries(cosmictiger_cuda_memory_pool_cuda_tests PRIVATE cosmictiger_cuda_memory_pool)
        add_test(NAME cosmictiger.cuda_memory_pool_cuda COMMAND cosmictiger_cuda_memory_pool_cuda_tests --hpx:threads=2 --hpx:bind=none)
        set_tests_properties(cosmictiger.cuda_memory_pool_cuda PROPERTIES PROCESSORS 1 TIMEOUT 120 SKIP_RETURN_CODE 77)
    endif()

    find_package(GTest CONFIG QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG f8d7d77c06936315286eb55f8de22cd23c188571 # v1.14.0
        )
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    add_executable(cosmictiger_units_tests tests/units.cpp)
    target_link_libraries(cosmictiger_units_tests PRIVATE cosmictiger_units GTest::gtest_main)
    target_compile_options(cosmictiger_units_tests PRIVATE -fno-fast-math)
    add_test(NAME cosmictiger.units COMMAND cosmictiger_units_tests)
    set_tests_properties(cosmictiger.units PROPERTIES PROCESSORS 1 TIMEOUT 30)

    add_executable(cosmictiger_healpix_tests tests/healpix/healpix_test.cpp)
    target_link_libraries(cosmictiger_healpix_tests PRIVATE cosmictiger_healpix GTest::gtest_main)
    target_compile_options(cosmictiger_healpix_tests PRIVATE -fno-fast-math -ffp-contract=off)
    add_test(NAME cosmictiger.healpix COMMAND cosmictiger_healpix_tests)
    set_tests_properties(cosmictiger.healpix PROPERTIES PROCESSORS 1 TIMEOUT 120)

    add_executable(cosmictiger_lightcone_tests tests/test_main.cpp tests/lightcone/lightcone.cpp)
    target_link_libraries(cosmictiger_lightcone_tests PRIVATE cosmictiger_lightcone GTest::gtest)
    target_compile_options(cosmictiger_lightcone_tests PRIVATE -fno-fast-math)
    add_test(NAME cosmictiger.lightcone COMMAND cosmictiger_lightcone_tests
        --hpx:threads=1 --hpx:bind=none --hpx:ini=hpx.threadpools.io_pool_size=1
        --hpx:ini=hpx.threadpools.timer_pool_size=1 --hpx:ini=hpx.threadpools.parcel_pool_size=1
        --hpx:ini=hpx.parcel.mpi.enable=0)
    set_tests_properties(cosmictiger.lightcone PROPERTIES PROCESSORS 6 TIMEOUT 120 RUN_SERIAL TRUE)

    add_executable(cosmictiger_evolution_tests tests/test_main.cpp tests/evolution/evolution_state.cpp
        tests/evolution/evolution_checkpoint.cpp tests/evolution/evolution.cpp tests/evolution/evolution_boundary.cpp
        tests/evolution/evolution_lightcone.cpp tests/gravity/kick/kick_sums.cpp tests/cartesian/cartesian_grid.cpp)
    target_link_libraries(cosmictiger_evolution_tests PRIVATE cosmictiger_evolution GTest::gtest)
    target_compile_options(cosmictiger_evolution_tests PRIVATE -fno-fast-math)
    add_test(NAME cosmictiger.evolution COMMAND cosmictiger_evolution_tests
        --hpx:threads=1 --hpx:bind=none --hpx:ini=hpx.threadpools.io_pool_size=1
        --hpx:ini=hpx.threadpools.timer_pool_size=1 --hpx:ini=hpx.threadpools.parcel_pool_size=1
        --hpx:ini=hpx.parcel.mpi.enable=0)
    set_tests_properties(cosmictiger.evolution PROPERTIES PROCESSORS 6 TIMEOUT 240 RUN_SERIAL TRUE)

    if(COSMICTIGER_ENABLE_CUDA)
        add_executable(cosmictiger_kick_cuda_tests tests/test_main.cpp tests/direct/direct_interactions_cuda.cu)
        target_link_libraries(cosmictiger_kick_cuda_tests PRIVATE cosmictiger_particles cosmictiger_direct_cuda GTest::gtest)
        add_test(NAME cosmictiger.kick_cuda COMMAND cosmictiger_kick_cuda_tests
            --hpx:threads=1 --hpx:bind=none --hpx:ini=hpx.threadpools.io_pool_size=1
            --hpx:ini=hpx.threadpools.timer_pool_size=1 --hpx:ini=hpx.threadpools.parcel_pool_size=1
            --hpx:ini=hpx.parcel.mpi.enable=0)
        set_tests_properties(cosmictiger.kick_cuda PROPERTIES PROCESSORS 8 TIMEOUT 120 RUN_SERIAL TRUE)

        add_executable(cosmictiger_cuda_warp_partition_tests tests/test_main.cpp tests/cuda/cuda_warp_partition.cu)
        target_link_libraries(cosmictiger_cuda_warp_partition_tests PRIVATE cosmictiger_dependencies GTest::gtest)
        add_test(NAME cosmictiger.cuda_warp_partition COMMAND cosmictiger_cuda_warp_partition_tests
            --hpx:threads=1 --hpx:bind=none --hpx:ini=hpx.threadpools.io_pool_size=1
            --hpx:ini=hpx.threadpools.timer_pool_size=1 --hpx:ini=hpx.threadpools.parcel_pool_size=1
            --hpx:ini=hpx.parcel.mpi.enable=0)
        set_tests_properties(cosmictiger.cuda_warp_partition PROPERTIES PROCESSORS 2 TIMEOUT 120 RUN_SERIAL TRUE)
    endif()

    add_executable(cosmictiger_debug_tests tests/debug.cpp)
    target_include_directories(cosmictiger_debug_tests PRIVATE "${PROJECT_SOURCE_DIR}")
    target_link_libraries(cosmictiger_debug_tests PRIVATE GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(cosmictiger_debug_tests PROPERTIES PROCESSORS 1 TIMEOUT 30)

    # Keep the counter isolated from production until instrumentation is added.
    add_executable(cosmictiger_flop_counter_tests tests/flop_counter/flop_counter.cpp)
    target_include_directories(cosmictiger_flop_counter_tests PRIVATE "${PROJECT_SOURCE_DIR}")
    target_link_libraries(cosmictiger_flop_counter_tests PRIVATE GTest::gtest_main)
    add_test(NAME cosmictiger.flop_counter COMMAND cosmictiger_flop_counter_tests)
    set_tests_properties(cosmictiger.flop_counter PROPERTIES PROCESSORS 1 TIMEOUT 30)

    add_executable(cosmictiger_flop_counter_contract_tests tests/flop_counter/flop_counter.cpp)
    target_include_directories(cosmictiger_flop_counter_contract_tests PRIVATE "${PROJECT_SOURCE_DIR}")
    target_link_libraries(cosmictiger_flop_counter_contract_tests PRIVATE GTest::gtest_main)
    target_compile_definitions(cosmictiger_flop_counter_contract_tests PRIVATE COSMICTIGER_VALIDATION_CHECKS=1)
    add_test(NAME cosmictiger.flop_counter_contracts COMMAND cosmictiger_flop_counter_contract_tests)
    set_tests_properties(cosmictiger.flop_counter_contracts PROPERTIES PROCESSORS 1 TIMEOUT 30)

    if(COSMICTIGER_ENABLE_CUDA)
        foreach(counterCheck cuda cuda_contracts)
            add_executable(cosmictiger_flop_counter_${counterCheck}_tests tests/test_main.cpp tests/flop_counter/flop_counter_cuda.cu)
            target_link_libraries(cosmictiger_flop_counter_${counterCheck}_tests PRIVATE cosmictiger_cuda_runtime GTest::gtest)
            if(counterCheck STREQUAL "cuda_contracts")
                target_compile_definitions(cosmictiger_flop_counter_${counterCheck}_tests PRIVATE COSMICTIGER_VALIDATION_CHECKS=1)
            endif()
            add_test(NAME cosmictiger.flop_counter_${counterCheck}
                COMMAND cosmictiger_flop_counter_${counterCheck}_tests --hpx:threads=1 --hpx:bind=none
                    --hpx:localities=1 --hpx:console --hpx:ini=hpx.parcel.enable=0
                    --hpx:ini=hpx.parcel.mpi.enable=0 --hpx:ini=hpx.parcel.tcp.enable=0
                    --hpx:ini=hpx.threadpools.io_pool_size=1 --hpx:ini=hpx.threadpools.timer_pool_size=1)
            set_tests_properties(cosmictiger.flop_counter_${counterCheck} PROPERTIES PROCESSORS 6 TIMEOUT 60 RUN_SERIAL TRUE)
        endforeach()
    endif()

    add_executable(cosmictiger_boltzmann_tests tests/test_main.cpp tests/einstein_boltzmann.cpp)
    target_link_libraries(cosmictiger_boltzmann_tests PRIVATE cosmictiger_boltzmann GTest::gtest)
    target_compile_options(cosmictiger_boltzmann_tests PRIVATE -fno-fast-math)
    add_test(NAME cosmictiger.boltzmann
        COMMAND cosmictiger_boltzmann_tests --hpx:threads=2 --hpx:bind=none)
    set_tests_properties(cosmictiger.boltzmann PROPERTIES PROCESSORS 4 TIMEOUT 180)

    add_executable(cosmictiger_initialization_tests tests/test_main.cpp tests/particle/particle_initialization.cpp)
    target_link_libraries(cosmictiger_initialization_tests PRIVATE cosmictiger_initialization GTest::gtest)
    target_compile_options(cosmictiger_initialization_tests PRIVATE -fno-fast-math)
    add_test(NAME cosmictiger.initialization
        COMMAND cosmictiger_initialization_tests --hpx:threads=1 --hpx:bind=none
            --hpx:ini=hpx.threadpools.io_pool_size=1 --hpx:ini=hpx.threadpools.timer_pool_size=1
            --hpx:ini=hpx.threadpools.parcel_pool_size=1 --hpx:ini=hpx.parcel.mpi.enable=0)
    add_test(NAME cosmictiger.initialization_parallel
        COMMAND cosmictiger_initialization_tests --hpx:threads=2 --hpx:bind=none
            --hpx:ini=hpx.threadpools.io_pool_size=1 --hpx:ini=hpx.threadpools.timer_pool_size=1
            --hpx:ini=hpx.threadpools.parcel_pool_size=1 --hpx:ini=hpx.parcel.mpi.enable=0)
    set_tests_properties(cosmictiger.initialization cosmictiger.initialization_parallel PROPERTIES PROCESSORS 6 TIMEOUT 180)

    add_executable(cosmictiger_initial_condition_silo_check tests/initial_condition/initial_condition_silo.cpp)
    target_link_libraries(cosmictiger_initial_condition_silo_check PRIVATE cosmictiger_particles HPX::hpx HPX::iostreams_component)
    target_compile_options(cosmictiger_initial_condition_silo_check PRIVATE -fno-fast-math)
    add_test(NAME cosmictiger.initial_condition_cli
        COMMAND "${CMAKE_COMMAND}"
            -DCLI_EXECUTABLE=$<TARGET_FILE:cosmictiger>
            -DVERIFY_EXECUTABLE=$<TARGET_FILE:cosmictiger_initial_condition_silo_check>
            -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/initial_condition_cli
            -P "${PROJECT_SOURCE_DIR}/tests/initial_condition/initial_condition_cli.cmake")
    set_tests_properties(cosmictiger.initial_condition_cli PROPERTIES PROCESSORS 6 TIMEOUT 360 RUN_SERIAL TRUE)

    add_executable(cosmictiger_particle_silo_tests tests/test_main.cpp tests/particle/particle_silo.cpp)
    target_link_libraries(cosmictiger_particle_silo_tests PRIVATE cosmictiger_particles GTest::gtest)
    add_test(NAME cosmictiger.particle_silo
        COMMAND cosmictiger_particle_silo_tests --hpx:threads=1 --hpx:bind=none
            --hpx:ini=hpx.threadpools.io_pool_size=1 --hpx:ini=hpx.threadpools.timer_pool_size=1
            --hpx:ini=hpx.threadpools.parcel_pool_size=1 --hpx:ini=hpx.parcel.mpi.enable=0)
    set_tests_properties(cosmictiger.particle_silo PROPERTIES PROCESSORS 6 TIMEOUT 180)

    add_executable(cosmictiger_symmetric_tests tests/test_main.cpp tests/symmetric/symmetric_operators.cpp tests/pmmm/plummer_green.cpp)
    target_include_directories(cosmictiger_symmetric_tests PRIVATE "${PROJECT_SOURCE_DIR}")
    target_link_libraries(cosmictiger_symmetric_tests PRIVATE HPX::hpx GTest::gtest)
    target_compile_options(cosmictiger_symmetric_tests PRIVATE -fno-fast-math)
    add_executable(cosmictiger_fmm_interactions_tests tests/test_main.cpp tests/fmm/fmm_interactions.cpp)
    if(COSMICTIGER_ENABLE_CUDA)
        target_sources(cosmictiger_fmm_interactions_tests PRIVATE tests/fmm/fmm_interactions_cuda.cu)
    endif()
    target_link_libraries(cosmictiger_fmm_interactions_tests PRIVATE cosmictiger_fmm_interactions GTest::gtest)
    target_compile_options(cosmictiger_fmm_interactions_tests PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>")
    add_test(NAME cosmictiger.fmm_interactions COMMAND cosmictiger_fmm_interactions_tests
        --hpx:threads=2 --hpx:bind=none --hpx:ini=hpx.threadpools.io_pool_size=1
        --hpx:ini=hpx.threadpools.timer_pool_size=1 --hpx:ini=hpx.threadpools.parcel_pool_size=1
        --hpx:ini=hpx.parcel.mpi.enable=0)
    set_tests_properties(cosmictiger.fmm_interactions PROPERTIES PROCESSORS 6 TIMEOUT 120 RUN_SERIAL TRUE)

    add_executable(cosmictiger_kd_tree_tests tests/test_main.cpp tests/tree/kd_tree.cpp)
    target_link_libraries(cosmictiger_kd_tree_tests PRIVATE cosmictiger_kd_tree GTest::gtest)
    target_compile_options(cosmictiger_kd_tree_tests PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>")
    add_test(NAME cosmictiger.kd_tree COMMAND cosmictiger_kd_tree_tests
        --hpx:threads=2 --hpx:bind=none --hpx:ini=hpx.threadpools.io_pool_size=1
        --hpx:ini=hpx.threadpools.timer_pool_size=1 --hpx:ini=hpx.threadpools.parcel_pool_size=1
        --hpx:ini=hpx.parcel.mpi.enable=0)
    set_tests_properties(cosmictiger.kd_tree PROPERTIES PROCESSORS 6 TIMEOUT 120 RUN_SERIAL TRUE)

    set(symmetricRuntimeOptions --hpx:threads=1 --hpx:bind=none --hpx:localities=1 --hpx:console
        --hpx:ini=hpx.parcel.enable=0 --hpx:ini=hpx.parcel.mpi.enable=0
        --hpx:ini=hpx.threadpools.io_pool_size=1 --hpx:ini=hpx.threadpools.timer_pool_size=1)
    if(HPX_WITH_PARCELPORT_TCP)
        list(APPEND symmetricRuntimeOptions --hpx:ini=hpx.parcel.tcp.enable=0)
    endif()
    add_test(NAME cosmictiger.symmetric COMMAND cosmictiger_symmetric_tests ${symmetricRuntimeOptions})
    set_tests_properties(cosmictiger.symmetric PROPERTIES PROCESSORS 6 TIMEOUT 180 RUN_SERIAL TRUE)

    add_executable(cosmictiger_tests src/fft/fft_hpx_exchange.cpp tests/test_main.cpp tests/dependencies.cpp tests/fft/fft.cpp tests/fft/fft_parallel.cpp tests/fft/fft_hpx_exchange.cpp
        tests/fft/fft_communication_plan.cpp tests/fixed.cpp tests/particle/particle_storage.cpp tests/cartesian/cartesian_grid.cpp tests/box.cpp tests/particle/particle_load_balance.cpp
        tests/particle/particle_ghost_exchange.cpp tests/particle/particle_silo.cpp tests/cell_multipoles.cpp tests/parallel.cpp tests/direct/direct_interactions.cpp tests/direct/direct_solve.cpp tests/direct/direct_comparison.cpp tests/direct/direct_pmmm.cpp
        tests/gravity/kick/kick.cpp tests/gravity/kick/kick_sums.cpp tests/gravity/kick/pmmm_kick.cpp
        tests/cartesian/cartesian_fmm.cpp tests/stf/stf_m2l.cpp tests/ferrers.cpp tests/pmmm/pmmm_green.cpp tests/pmmm/pmmm_contraction.cpp tests/pmmm/pmmm_convolution.cpp tests/pmmm/plummer_convolution.cpp tests/gravity/kick/plummer_kick.cpp)
    if(COSMICTIGER_ENABLE_CUDA)
        target_sources(cosmictiger_tests PRIVATE tests/ferrers_cuda.cu tests/direct/direct_interactions_cuda.cu tests/direct/direct_solve_cuda.cu
            tests/cuda/cuda_runtime.cpp tests/cuda/cuda_runtime_cuda.cu tests/stf/stf_m2p_cuda.cu tests/symmetric/symmetric_cuda.cu tests/cartesian/p2m_cuda.cpp)
    endif()
    set_source_files_properties(tests/direct/direct_solve.cpp tests/direct/direct_comparison.cpp tests/direct/direct_pmmm.cpp tests/stf/stf_m2l.cpp tests/pmmm/pmmm_green.cpp tests/pmmm/pmmm_contraction.cpp tests/pmmm/pmmm_convolution.cpp tests/pmmm/plummer_convolution.cpp tests/gravity/kick/plummer_kick.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math")
    target_link_libraries(cosmictiger_tests PRIVATE cosmictiger_dependencies cosmictiger_fmm cosmictiger_particles cosmictiger_kick cosmictiger_direct_solve cosmictiger_pmmm GTest::gtest)
    if(COSMICTIGER_ENABLE_CUDA)
        target_link_libraries(cosmictiger_tests PRIVATE cosmictiger_direct_cuda cosmictiger_stf_m2p_cuda)
    endif()
    include(GoogleTest)
    gtest_discover_tests(cosmictiger_tests PROPERTIES PROCESSORS 2 TIMEOUT 30)

    # Exercise header-only kick contracts with IEEE traps even in a Release
    # application build, without changing the installed HPX build mode.
    add_executable(cosmictiger_kick_contract_tests tests/test_main.cpp tests/gravity/kick/kick.cpp)
    target_include_directories(cosmictiger_kick_contract_tests PRIVATE "${PROJECT_SOURCE_DIR}")
    target_link_libraries(cosmictiger_kick_contract_tests PRIVATE HPX::hpx GTest::gtest)
    target_compile_definitions(cosmictiger_kick_contract_tests PRIVATE COSMICTIGER_VALIDATION_CHECKS=1)
    target_compile_options(cosmictiger_kick_contract_tests PRIVATE -fno-fast-math -ftrapping-math)
    target_link_options(cosmictiger_kick_contract_tests PRIVATE -fno-fast-math)
    add_test(NAME cosmictiger.kick_contracts COMMAND cosmictiger_kick_contract_tests
        --hpx:threads=1 --hpx:bind=none --hpx:ini=hpx.threadpools.io_pool_size=1
        --hpx:ini=hpx.threadpools.timer_pool_size=1 --hpx:ini=hpx.threadpools.parcel_pool_size=1
        --hpx:ini=hpx.parcel.mpi.enable=0)
    set_tests_properties(cosmictiger.kick_contracts PROPERTIES PROCESSORS 6 TIMEOUT 120)

    add_executable(cosmictiger_drift_tests tests/test_main.cpp tests/drift/drift.cpp)
    target_link_libraries(cosmictiger_drift_tests PRIVATE cosmictiger_drift cosmictiger_cosmology GTest::gtest)
    if(COSMICTIGER_ENABLE_CUDA)
        target_sources(cosmictiger_drift_tests PRIVATE tests/drift/drift_cuda.cu)
    endif()
    add_test(NAME cosmictiger.drift COMMAND cosmictiger_drift_tests
        --hpx:threads=2 --hpx:bind=none --hpx:localities=1 --hpx:console
        --hpx:ini=hpx.parcel.enable=0 --hpx:ini=hpx.parcel.mpi.enable=0 --hpx:ini=hpx.parcel.tcp.enable=0
        --hpx:ini=hpx.threadpools.io_pool_size=1 --hpx:ini=hpx.threadpools.timer_pool_size=1)
    set_tests_properties(cosmictiger.drift PROPERTIES PROCESSORS 8 TIMEOUT 60 RUN_SERIAL TRUE)

    # Compile the drift implementation with IEEE traps and contracts, including
    # in Release builds, without requiring another installed HPX build mode.
    add_executable(cosmictiger_drift_contract_tests tests/test_main.cpp tests/drift/drift.cpp src/drift/drift.cpp)
    target_link_libraries(cosmictiger_drift_contract_tests PRIVATE cosmictiger_particles cosmictiger_cosmology GTest::gtest)
    target_compile_definitions(cosmictiger_drift_contract_tests PRIVATE COSMICTIGER_VALIDATION_CHECKS=1)
    target_compile_options(cosmictiger_drift_contract_tests PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math;-ftrapping-math>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-fast-math;-Xcompiler=-ftrapping-math>")
    target_link_options(cosmictiger_drift_contract_tests PRIVATE -fno-fast-math)
    if(COSMICTIGER_ENABLE_CUDA)
        target_sources(cosmictiger_drift_contract_tests PRIVATE src/drift/drift_cuda.cu tests/drift/drift_cuda.cu)
    endif()
    add_test(NAME cosmictiger.drift_contracts COMMAND cosmictiger_drift_contract_tests
        --hpx:threads=1 --hpx:bind=none --hpx:localities=1 --hpx:console
        --hpx:ini=hpx.parcel.enable=0 --hpx:ini=hpx.parcel.mpi.enable=0 --hpx:ini=hpx.parcel.tcp.enable=0
        --hpx:ini=hpx.threadpools.io_pool_size=1 --hpx:ini=hpx.threadpools.timer_pool_size=1)
    set_tests_properties(cosmictiger.drift_contracts PROPERTIES PROCESSORS 6 TIMEOUT 60 RUN_SERIAL TRUE)

    add_executable(cosmictiger_friedmann_tests tests/friedmann.cpp tests/simulation_time.cpp tests/evolution/evolution_end.cpp)
    target_link_libraries(cosmictiger_friedmann_tests PRIVATE cosmictiger_cosmology GTest::gtest_main)
    target_compile_options(cosmictiger_friedmann_tests PRIVATE -fno-fast-math)
    gtest_discover_tests(cosmictiger_friedmann_tests PROPERTIES PROCESSORS 1 TIMEOUT 30)

    foreach(stfCheck smoke operators)
        add_executable(cosmictiger_stf_${stfCheck} tests/stf/stf_${stfCheck}.cpp)
        target_link_libraries(cosmictiger_stf_${stfCheck} PRIVATE cosmictiger_stf)
        # Independent oracles include nonfinite-input checks, which require
        # IEEE arithmetic even when the application uses -Ofast.
        target_compile_options(cosmictiger_stf_${stfCheck} PRIVATE -fno-fast-math)
        add_test(NAME cosmictiger.stf_${stfCheck} COMMAND cosmictiger_stf_${stfCheck})
        set_tests_properties(cosmictiger.stf_${stfCheck} PROPERTIES PROCESSORS 1 TIMEOUT 60)
    endforeach()

    add_test(NAME cosmictiger.fft_single_worker
        COMMAND cosmictiger_tests --gtest_filter=Fft* --hpx:threads=1)
    set_tests_properties(cosmictiger.fft_single_worker PROPERTIES PROCESSORS 1 TIMEOUT 30)

    add_test(NAME cosmictiger.pmmm_single_worker
        COMMAND cosmictiger_tests --gtest_filter=Pmmm* --hpx:threads=1)
    set_tests_properties(cosmictiger.pmmm_single_worker PROPERTIES PROCESSORS 1 TIMEOUT 180)

    add_test(NAME cosmictiger.gravity_single_worker
        COMMAND cosmictiger_tests --gtest_filter=CartesianFmm*:DirectInteractionsCpu*:ParallelChunks*
            --hpx:threads=1 --hpx:bind=none --hpx:localities=1 --hpx:console
            --hpx:ini=hpx.parcel.enable=0 --hpx:ini=hpx.parcel.mpi.enable=0 --hpx:ini=hpx.parcel.tcp.enable=0
            --hpx:ini=hpx.threadpools.io_pool_size=1 --hpx:ini=hpx.threadpools.timer_pool_size=1)
    set_tests_properties(cosmictiger.gravity_single_worker PROPERTIES PROCESSORS 6 TIMEOUT 30 RUN_SERIAL TRUE)

    add_executable(cosmictiger_global_constants_tests tests/global_constants.cpp)
    target_link_libraries(cosmictiger_global_constants_tests PRIVATE cosmictiger_global_constants HPX::hpx HPX::iostreams_component)
    add_test(NAME cosmictiger.global_constants
        COMMAND cosmictiger_global_constants_tests --hpx:threads=2 --hpx:bind=none)
    set_tests_properties(cosmictiger.global_constants PROPERTIES PROCESSORS 2 TIMEOUT 30)

    add_test(NAME cosmictiger.startup COMMAND cosmictiger --simulation.evolve=false --hpx:threads=2)
    set_tests_properties(cosmictiger.startup PROPERTIES
        PROCESSORS 2
        TIMEOUT 30
    )

    set(mpiRuntimeOptions --hpx:threads=2 --hpx:bind=none --hpx:ini=hpx.parcel.mpi.enable=1)
    if(HPX_WITH_PARCELPORT_TCP)
        list(APPEND mpiRuntimeOptions --hpx:ini=hpx.parcel.tcp.enable=0)
    endif()
    add_test(NAME cosmictiger.mpi_startup
        COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cosmictiger> ${MPIEXEC_POSTFLAGS}
            ${mpiRuntimeOptions} --simulation.evolve=false
    )
    set_tests_properties(cosmictiger.mpi_startup PROPERTIES
        PROCESSORS 4
        TIMEOUT 30
    )

    add_test(NAME cosmictiger.global_constants_remote
        COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cosmictiger_global_constants_tests> ${MPIEXEC_POSTFLAGS}
            ${mpiRuntimeOptions})
    set_tests_properties(cosmictiger.global_constants_remote PROPERTIES PROCESSORS 4 TIMEOUT 30)

    add_test(NAME cosmictiger.direct_comparison_remote
        COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cosmictiger_tests> ${MPIEXEC_POSTFLAGS}
            --gtest_filter=DirectComparison.RemoteOwnedSources ${mpiRuntimeOptions})
    set_tests_properties(cosmictiger.direct_comparison_remote PROPERTIES PROCESSORS 4 TIMEOUT 60)

    add_test(NAME cosmictiger.fft_hpx_remote
        COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cosmictiger_tests> ${MPIEXEC_POSTFLAGS}
            --gtest_filter=FftHpxExchange.RemoteReferencePayload ${mpiRuntimeOptions}
    )
    set_tests_properties(cosmictiger.fft_hpx_remote PROPERTIES
        PROCESSORS 4
        TIMEOUT 30
    )

    # Eight logical FFT endpoints share two physical processes, exercising
    # coefficient batches through real MPI transport without eight runtimes.
    set(pmmmMpiRuntimeOptions --hpx:threads=1 --hpx:bind=none
        --hpx:ini=hpx.threadpools.io_pool_size=1 --hpx:ini=hpx.threadpools.timer_pool_size=1
        --hpx:ini=hpx.threadpools.parcel_pool_size=1 --hpx:ini=hpx.parcel.mpi.enable=1)
    if(HPX_WITH_PARCELPORT_TCP)
        list(APPEND pmmmMpiRuntimeOptions --hpx:ini=hpx.parcel.tcp.enable=0)
    endif()
    add_test(NAME cosmictiger.pmmm_remote
        COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cosmictiger_tests> ${MPIEXEC_POSTFLAGS}
            --gtest_filter=PmmmConvolutionTest.RemoteCoefficientBatches ${pmmmMpiRuntimeOptions})
    set_tests_properties(cosmictiger.pmmm_remote PROPERTIES PROCESSORS 16 TIMEOUT 180 RUN_SERIAL TRUE)

    add_test(NAME cosmictiger.kick_sums_remote
        COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cosmictiger_tests> ${MPIEXEC_POSTFLAGS}
            --gtest_filter=KickSums.PhysicalLocalityReduction:KickSums.PotentialSelectionMismatchReachesEveryPhysicalLocality
            ${pmmmMpiRuntimeOptions})
    set_tests_properties(cosmictiger.kick_sums_remote PROPERTIES PROCESSORS 16 TIMEOUT 60 RUN_SERIAL TRUE)

    add_test(NAME cosmictiger.particle_load_balance_single_worker
        COMMAND cosmictiger_tests --gtest_filter=ParticleLoadBalanceTest.* --hpx:threads=1)
    set_tests_properties(cosmictiger.particle_load_balance_single_worker PROPERTIES PROCESSORS 1 TIMEOUT 30)

    add_test(NAME cosmictiger.particle_ghost_exchange_single_worker
        COMMAND cosmictiger_tests --gtest_filter=ParticleGhostExchangeTest.* --hpx:threads=1)
    set_tests_properties(cosmictiger.particle_ghost_exchange_single_worker PROPERTIES PROCESSORS 1 TIMEOUT 30)

    # Eight logical grid localities on two processes exercise remote particle
    # transfers while remaining within the conservative local CPU budget.
    add_test(NAME cosmictiger.particle_ghost_exchange_remote
        COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cosmictiger_tests> ${MPIEXEC_POSTFLAGS}
            --gtest_filter=ParticleGhostExchangeTest.RemoteExchange ${mpiRuntimeOptions})
    set_tests_properties(cosmictiger.particle_ghost_exchange_remote PROPERTIES PROCESSORS 4 TIMEOUT 60)

    # Explicit opt-in: cubic decompositions need at least eight localities.
    # This cannot run within the six-core budget while War Thunder is running.
    option(COSMICTIGER_TEST_EIGHT_LOCALITIES "Enable the eight-locality particle redistribution integration test" OFF)
    if(COSMICTIGER_TEST_EIGHT_LOCALITIES)
        add_test(NAME cosmictiger.particle_load_balance_remote
            COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} 8
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cosmictiger_tests> ${MPIEXEC_POSTFLAGS}
                --gtest_filter=ParticleLoadBalanceTest.RemoteRedistribution --hpx:threads=1
                --hpx:bind=none --hpx:ini=hpx.parcel.mpi.enable=1 --hpx:ini=hpx.parcel.tcp.enable=0)
        set_tests_properties(cosmictiger.particle_load_balance_remote PROPERTIES PROCESSORS 8 TIMEOUT 60)

        add_test(NAME cosmictiger.particle_ghost_exchange_eight_localities
            COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} 8
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cosmictiger_tests> ${MPIEXEC_POSTFLAGS}
                --gtest_filter=ParticleGhostExchangeTest.RemoteExchange --hpx:threads=1
                --hpx:bind=none --hpx:ini=hpx.parcel.mpi.enable=1 --hpx:ini=hpx.parcel.tcp.enable=0)
        set_tests_properties(cosmictiger.particle_ghost_exchange_eight_localities PROPERTIES PROCESSORS 8 TIMEOUT 60)
    endif()
endif()

# Cover every root-directory project target, including standalone header tests.
# The profiler itself depends only on HPX and platform libraries, so attaching
# it to units/STF and the other public interfaces cannot create a target cycle.
get_property(cosmictigerProfileTargets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
foreach(cosmictigerProfileTarget IN LISTS cosmictigerProfileTargets)
    if(cosmictigerProfileTarget STREQUAL "cosmictiger_profiler")
        continue()
    endif()
    get_target_property(cosmictigerProfileTargetType "${cosmictigerProfileTarget}" TYPE)
    if(cosmictigerProfileTargetType STREQUAL "INTERFACE_LIBRARY")
        target_link_libraries("${cosmictigerProfileTarget}" INTERFACE cosmictiger_profiler)
    elseif(cosmictigerProfileTargetType MATCHES "^(STATIC_LIBRARY|SHARED_LIBRARY|MODULE_LIBRARY)$")
        target_link_libraries("${cosmictigerProfileTarget}" PUBLIC cosmictiger_profiler)
    elseif(cosmictigerProfileTargetType STREQUAL "EXECUTABLE")
        target_link_libraries("${cosmictigerProfileTarget}" PRIVATE cosmictiger_profiler)
    endif()
endforeach()
