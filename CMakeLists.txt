cmake_minimum_required(VERSION 3.25)
project(cosmictiger VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(COSMICTIGER_ENABLE_CUDA "Enable CUDA compilation and runtime support" ON)

find_package(HPX CONFIG REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)
if(NOT HPX_WITH_NETWORKING OR NOT HPX_WITH_PARCELPORT_MPI)
    message(FATAL_ERROR "CosmicTiger requires HPX built with networking and the MPI parcelport enabled.")
endif()
find_package(PkgConfig REQUIRED)
pkg_check_modules(SILO REQUIRED IMPORTED_TARGET silo)

if(COSMICTIGER_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

add_library(cosmictiger_dependencies INTERFACE)
target_include_directories(cosmictiger_dependencies INTERFACE "${PROJECT_SOURCE_DIR}")
target_link_libraries(cosmictiger_dependencies INTERFACE HPX::hpx MPI::MPI_CXX PkgConfig::SILO)
if(COSMICTIGER_ENABLE_CUDA)
    target_link_libraries(cosmictiger_dependencies INTERFACE CUDA::cudart)
    target_compile_definitions(cosmictiger_dependencies INTERFACE COSMICTIGER_ENABLE_CUDA=1)
endif()

add_executable(cosmictiger src/main.cpp)
target_link_libraries(cosmictiger PRIVATE cosmictiger_dependencies)
target_compile_options(cosmictiger PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic>")

include(CTest)
if(BUILD_TESTING)
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

    add_executable(cosmictiger_tests tests/dependencies.cpp)
    target_link_libraries(cosmictiger_tests PRIVATE cosmictiger_dependencies GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(cosmictiger_tests)

    add_test(NAME cosmictiger.startup COMMAND cosmictiger --hpx:threads=2)
    set_tests_properties(cosmictiger.startup PROPERTIES
        PROCESSORS 2
        TIMEOUT 30
    )

    add_test(NAME cosmictiger.mpi_startup
        COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cosmictiger> ${MPIEXEC_POSTFLAGS}
            --hpx:threads=2 --hpx:bind=none
            --hpx:ini=hpx.parcel.mpi.enable=1 --hpx:ini=hpx.parcel.tcp.enable=0
    )
    set_tests_properties(cosmictiger.mpi_startup PROPERTIES
        PROCESSORS 4
        TIMEOUT 30
    )
endif()
