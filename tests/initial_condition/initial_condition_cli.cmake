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

foreach(requiredVariable CLI_EXECUTABLE VERIFY_EXECUTABLE OUTPUT_DIR)
    if(NOT DEFINED ${requiredVariable} OR "${${requiredVariable}}" STREQUAL "")
        message(FATAL_ERROR "${requiredVariable} is required for the initial-condition CLI test.")
    endif()
endforeach()

file(MAKE_DIRECTORY "${OUTPUT_DIR}")
set(outputPrefix "${OUTPUT_DIR}/initial_conditions")
set(outputFile "${outputPrefix}.locality0.silo")
set(outputVisit "${outputPrefix}.visit")
file(REMOVE "${outputFile}" "${outputVisit}")

execute_process(
    COMMAND "${CLI_EXECUTABLE}"
        --initialConditions.generate
        --simulation.evolve=false
        "--initialConditions.outputPrefix=${outputPrefix}"
        --initialConditions.meshExponent=2
        --initialConditions.boxLength=100
        --initialConditions.seed=1234
        --initialConditions.maximumDensityDeviation=0.05
        --initialConditions.earliestScaleFactor=0.002
        --initialConditions.maximumLogScaleFactorStep=0.5
        --initialConditions.logScaleFactorTolerance=0.001
        --hpx:threads=1
        --hpx:bind=none
        --hpx:ini=hpx.threadpools.io_pool_size=1
        --hpx:ini=hpx.threadpools.timer_pool_size=1
        --hpx:ini=hpx.threadpools.parcel_pool_size=1
        --hpx:ini=hpx.parcel.mpi.enable=0
    RESULT_VARIABLE cliResult
    OUTPUT_VARIABLE cliOutput
    ERROR_VARIABLE cliError
    TIMEOUT 150
)
file(WRITE "${OUTPUT_DIR}/initial_conditions.log" "${cliOutput}\n${cliError}")
if(NOT "${cliResult}" STREQUAL "0")
    message(FATAL_ERROR "Initial-condition CLI failed (${cliResult}):\n${cliOutput}\n${cliError}")
endif()
if(NOT cliOutput MATCHES "Initial conditions written to")
    message(FATAL_ERROR "Initial-condition CLI omitted its success report:\n${cliOutput}\n${cliError}")
endif()
if(NOT EXISTS "${outputFile}")
    message(FATAL_ERROR "Initial-condition CLI did not create ${outputFile}.")
endif()
if(NOT EXISTS "${outputVisit}")
    message(FATAL_ERROR "Initial-condition CLI did not create ${outputVisit}.")
endif()
file(READ "${outputVisit}" visitContents)
if(NOT visitContents STREQUAL "initial_conditions.locality0.silo\n")
    message(FATAL_ERROR "Initial-condition VisIt series has unexpected contents: ${visitContents}")
endif()

execute_process(
    COMMAND "${VERIFY_EXECUTABLE}" "${outputFile}"
    RESULT_VARIABLE verifyResult
    OUTPUT_VARIABLE verifyOutput
    ERROR_VARIABLE verifyError
    TIMEOUT 30
)
if(NOT "${verifyResult}" STREQUAL "0")
    message(FATAL_ERROR "Initial-condition Silo verification failed (${verifyResult}):\n${verifyOutput}\n${verifyError}")
endif()
message(STATUS "${verifyOutput}")

# Initial output must contain computed fields without advancing any particle rung.
set(gravityOutputPrefix "${OUTPUT_DIR}/initial_conditions_gravity")
set(gravityOutputFile "${gravityOutputPrefix}.locality0.silo")
file(REMOVE "${gravityOutputFile}")
execute_process(
    COMMAND "${CLI_EXECUTABLE}"
        --initialConditions.generate
        --simulation.evolve=false
        "--initialConditions.outputPrefix=${gravityOutputPrefix}"
        --initialConditions.meshExponent=3
        --gravity.meshExponent=2
        --gravity.ghostWidth=1
        --initialConditions.boxLength=100
        --initialConditions.seed=1234
        --initialConditions.maximumDensityDeviation=0.05
        --initialConditions.earliestScaleFactor=0.002
        --initialConditions.maximumLogScaleFactorStep=0.5
        --initialConditions.logScaleFactorTolerance=0.001
        --output.acceleration
        --output.potential
        --hpx:threads=1
        --hpx:bind=none
        --hpx:ini=hpx.threadpools.io_pool_size=1
        --hpx:ini=hpx.threadpools.timer_pool_size=1
        --hpx:ini=hpx.threadpools.parcel_pool_size=1
        --hpx:ini=hpx.parcel.mpi.enable=0
    RESULT_VARIABLE gravityResult
    OUTPUT_VARIABLE gravityOutput
    ERROR_VARIABLE gravityError
    TIMEOUT 150
)
file(WRITE "${OUTPUT_DIR}/initial_conditions_gravity.log" "${gravityOutput}\n${gravityError}")
if(NOT "${gravityResult}" STREQUAL "0")
    message(FATAL_ERROR "Initial-condition gravity CLI failed (${gravityResult}):\n${gravityOutput}\n${gravityError}")
endif()
if(NOT gravityOutput MATCHES "Initial conditions written to" OR
   NOT gravityOutput MATCHES "Global force sum" OR NOT gravityOutput MATCHES "Global potential sum")
    message(FATAL_ERROR "Initial-condition gravity CLI omitted its output or global-sum report:\n${gravityOutput}\n${gravityError}")
endif()
if(NOT EXISTS "${gravityOutputFile}")
    message(FATAL_ERROR "Initial-condition gravity CLI did not create ${gravityOutputFile}.")
endif()
execute_process(
    COMMAND "${VERIFY_EXECUTABLE}" "${gravityOutputFile}" --independent-grid-fields
    RESULT_VARIABLE gravityVerifyResult
    OUTPUT_VARIABLE gravityVerifyOutput
    ERROR_VARIABLE gravityVerifyError
    TIMEOUT 30
)
if(NOT "${gravityVerifyResult}" STREQUAL "0")
    message(FATAL_ERROR "Initial-condition gravity Silo verification failed (${gravityVerifyResult}):\n${gravityVerifyOutput}\n${gravityVerifyError}")
endif()
message(STATUS "${gravityVerifyOutput}")

# Reject a population with no physical density before running the EB solver.
execute_process(
    COMMAND "${CLI_EXECUTABLE}"
        --initialConditions.generate
        --simulation.evolve=false
        "--initialConditions.outputPrefix=${OUTPUT_DIR}/invalid_density"
        --initialConditions.component=cdm
        --cosmology.omegaCdmH2=0
        --hpx:threads=1 --hpx:bind=none
        --hpx:ini=hpx.threadpools.io_pool_size=1
        --hpx:ini=hpx.threadpools.timer_pool_size=1
        --hpx:ini=hpx.threadpools.parcel_pool_size=1
        --hpx:ini=hpx.parcel.mpi.enable=0
    RESULT_VARIABLE invalidResult
    OUTPUT_VARIABLE invalidOutput
    ERROR_VARIABLE invalidError
    TIMEOUT 30
)
if(NOT "${invalidResult}" STREQUAL "1" OR NOT invalidError MATCHES "must have positive physical matter density")
    message(FATAL_ERROR "Zero-density particle initialization was not rejected early:\n${invalidOutput}\n${invalidError}")
endif()
if(EXISTS "${OUTPUT_DIR}/invalid_density.locality0.silo")
    message(FATAL_ERROR "Rejected initialization wrote a Silo file.")
endif()
