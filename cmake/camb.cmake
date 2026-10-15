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

# The same file implements the configure-time target and its build-time script.
# Upstream's native Makefiles write into their source directory, so compile a
# private copy for each CosmicTiger build mode and leave the submodule clean.
if(COSMICTIGER_BUILD_CAMB)
    foreach(requiredVariable cosmictigerCambSourceDir cosmictigerCambStageDir
            COSMICTIGER_CAMB_MAKE_EXECUTABLE COSMICTIGER_CAMB_FORTRAN_COMPILER)
        if(NOT DEFINED ${requiredVariable} OR "${${requiredVariable}}" STREQUAL "")
            message(FATAL_ERROR "The CAMB build script requires ${requiredVariable}.")
        endif()
    endforeach()

    file(GLOB cambSourceFiles LIST_DIRECTORIES false
        "${cosmictigerCambSourceDir}/fortran/*.f90"
        "${cosmictigerCambSourceDir}/fortran/Makefile*"
        "${cosmictigerCambSourceDir}/fortran/*.dat"
        "${cosmictigerCambSourceDir}/forutils/*.f90"
        "${cosmictigerCambSourceDir}/forutils/Makefile*"
        "${cosmictigerCambSourceDir}/forutils/LICENSE"
        "${cosmictigerCambSourceDir}/camb/*.dat"
        "${cosmictigerCambSourceDir}/LICENCE.txt"
        "${cosmictigerCambSourceDir}/.gitmodules")
    list(SORT cambSourceFiles)
    set(sourceSignature "${COSMICTIGER_CAMB_FORTRAN_COMPILER}\n")
    execute_process(COMMAND "${COSMICTIGER_CAMB_FORTRAN_COMPILER}" --version
        OUTPUT_VARIABLE compilerVersion RESULT_VARIABLE compilerResult)
    if(NOT compilerResult EQUAL 0)
        message(FATAL_ERROR "Cannot run the CAMB Fortran compiler: ${COSMICTIGER_CAMB_FORTRAN_COMPILER}")
    endif()
    string(APPEND sourceSignature "${compilerVersion}\n")
    file(SHA256 "${CMAKE_CURRENT_LIST_FILE}" buildScriptHash)
    string(APPEND sourceSignature "${buildScriptHash}\n")
    foreach(sourceFile IN LISTS cambSourceFiles)
        file(RELATIVE_PATH sourceRelativePath "${cosmictigerCambSourceDir}" "${sourceFile}")
        file(SHA256 "${sourceFile}" sourceHash)
        string(APPEND sourceSignature "${sourceRelativePath} ${sourceHash}\n")
    endforeach()
    string(SHA256 sourceSignature "${sourceSignature}")
    set(signatureFile "${cosmictigerCambStageDir}/.cosmictiger_source_signature")
    set(previousSignature "")
    if(EXISTS "${signatureFile}")
        file(READ "${signatureFile}" previousSignature)
    endif()
    if(NOT sourceSignature STREQUAL previousSignature)
        if(NOT EXISTS "${signatureFile}")
            # Generators may pre-create output directories before this script runs.
            # Refuse to erase actual files without an ownership marker.
            file(GLOB_RECURSE unownedStageFiles LIST_DIRECTORIES false
                "${cosmictigerCambStageDir}/fortran/*"
                "${cosmictigerCambStageDir}/forutils/*"
                "${cosmictigerCambStageDir}/camb/*")
            if(unownedStageFiles)
                message(FATAL_ERROR "CAMB staging directory has no CosmicTiger ownership marker: ${cosmictigerCambStageDir}")
            endif()
        endif()
        file(REMOVE_RECURSE "${cosmictigerCambStageDir}/fortran"
            "${cosmictigerCambStageDir}/forutils" "${cosmictigerCambStageDir}/camb")
        foreach(sourceFile IN LISTS cambSourceFiles)
            file(RELATIVE_PATH sourceRelativePath "${cosmictigerCambSourceDir}" "${sourceFile}")
            get_filename_component(sourceParent "${sourceRelativePath}" DIRECTORY)
            file(COPY "${sourceFile}" DESTINATION "${cosmictigerCambStageDir}/${sourceParent}")
        endforeach()
        file(WRITE "${signatureFile}" "${sourceSignature}")
    endif()

    # Use the upstream optimized native solver in every application build mode.
    # A serial nested make contributes one compile job to the outer build budget.
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env
        --unset=MAKEFLAGS --unset=MFLAGS --unset=CAMB_BUILD_DIR
        --unset=FORUTILS_BUILD_DIR --unset=FORUTILSPATH
        --unset=OUTPUT_DIR --unset=DLL_DIR
        "${COSMICTIGER_CAMB_MAKE_EXECUTABLE}" --no-print-directory -j1
        COMPILER=gfortran "F90C=${COSMICTIGER_CAMB_FORTRAN_COMPILER}" camb
        WORKING_DIRECTORY "${cosmictigerCambStageDir}/fortran"
        RESULT_VARIABLE buildResult)
    if(NOT buildResult EQUAL 0)
        message(FATAL_ERROR "CAMB native solver build failed (${buildResult}).")
    endif()
    return()
endif()

set(cosmictigerCambSourceDir "${PROJECT_SOURCE_DIR}/third_party/camb")
if(NOT EXISTS "${cosmictigerCambSourceDir}/fortran/Makefile" OR
        NOT EXISTS "${cosmictigerCambSourceDir}/forutils/Makefile")
    message(FATAL_ERROR "CAMB and its forutils submodule are required. Run ./build.sh, or run: git submodule update --init --recursive -- third_party/camb")
endif()
find_program(COSMICTIGER_CAMB_MAKE_EXECUTABLE NAMES gmake make REQUIRED)
find_program(COSMICTIGER_CAMB_FORTRAN_COMPILER NAMES gfortran REQUIRED)
execute_process(COMMAND "${COSMICTIGER_CAMB_MAKE_EXECUTABLE}" --version
    OUTPUT_VARIABLE cambMakeVersion RESULT_VARIABLE cambMakeResult)
if(NOT cambMakeResult EQUAL 0 OR NOT cambMakeVersion MATCHES "GNU Make")
    message(FATAL_ERROR "CAMB's upstream build requires GNU Make.")
endif()
set(cosmictigerCambStageDir "${PROJECT_BINARY_DIR}/_deps/camb-src")
set(COSMICTIGER_CAMB_EXECUTABLE "${cosmictigerCambStageDir}/fortran/camb")
file(GLOB cambBuildInputs CONFIGURE_DEPENDS LIST_DIRECTORIES false
    "${cosmictigerCambSourceDir}/fortran/*.f90"
    "${cosmictigerCambSourceDir}/fortran/Makefile*"
    "${cosmictigerCambSourceDir}/fortran/*.dat"
    "${cosmictigerCambSourceDir}/forutils/*.f90"
    "${cosmictigerCambSourceDir}/forutils/Makefile*"
    "${cosmictigerCambSourceDir}/forutils/LICENSE"
    "${cosmictigerCambSourceDir}/camb/*.dat"
    "${cosmictigerCambSourceDir}/LICENCE.txt"
    "${cosmictigerCambSourceDir}/.gitmodules")
add_custom_command(OUTPUT "${COSMICTIGER_CAMB_EXECUTABLE}"
    COMMAND "${CMAKE_COMMAND}" -DCOSMICTIGER_BUILD_CAMB=ON
        "-DcosmictigerCambSourceDir=${cosmictigerCambSourceDir}"
        "-DcosmictigerCambStageDir=${cosmictigerCambStageDir}"
        "-DCOSMICTIGER_CAMB_MAKE_EXECUTABLE=${COSMICTIGER_CAMB_MAKE_EXECUTABLE}"
        "-DCOSMICTIGER_CAMB_FORTRAN_COMPILER=${COSMICTIGER_CAMB_FORTRAN_COMPILER}"
        -P "${CMAKE_CURRENT_LIST_FILE}"
    DEPENDS ${cambBuildInputs} "${CMAKE_CURRENT_LIST_FILE}"
        "${COSMICTIGER_CAMB_FORTRAN_COMPILER}"
    BYPRODUCTS "${cosmictigerCambStageDir}/.cosmictiger_source_signature"
    COMMENT "Building the CAMB native Einstein-Boltzmann solver"
    VERBATIM)
add_custom_target(cosmictiger_camb_solver DEPENDS "${COSMICTIGER_CAMB_EXECUTABLE}")
